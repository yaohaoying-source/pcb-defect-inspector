#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
} CpuStat;

typedef struct {
    double cpu_percent;
    long mem_total_mb;
    long mem_avail_mb;
    double mem_used_percent;
    long swap_total_mb;
    long swap_free_mb;
    double swap_used_percent;
    double temp_c;
    long cpu_mhz;
    long npu_mhz;
    double disk_free_gb;
    double disk_used_percent;
} PerfData;

static int read_cpu(CpuStat *s) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return -1;
    int ok = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                    &s->user, &s->nice, &s->system, &s->idle,
                    &s->iowait, &s->irq, &s->softirq, &s->steal);
    fclose(f);
    return ok >= 4 ? 0 : -1;
}

static unsigned long long cpu_total(const CpuStat *s) {
    return s->user + s->nice + s->system + s->idle + s->iowait + s->irq + s->softirq + s->steal;
}

static double calc_cpu(const CpuStat *prev, const CpuStat *cur) {
    unsigned long long prev_idle = prev->idle + prev->iowait;
    unsigned long long cur_idle = cur->idle + cur->iowait;
    unsigned long long total_delta = cpu_total(cur) - cpu_total(prev);
    unsigned long long idle_delta = cur_idle - prev_idle;
    if (total_delta == 0) return 0.0;
    return (double)(total_delta - idle_delta) * 100.0 / (double)total_delta;
}

static long read_long_file(const char *path) {
    FILE *f = fopen(path, "r");
    long v = -1;
    if (!f) return -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static void read_mem(PerfData *d) {
    FILE *f = fopen("/proc/meminfo", "r");
    char key[64], unit[16];
    long value;
    long mem_total = 0, mem_avail = 0, swap_total = 0, swap_free = 0;
    if (!f) return;
    while (fscanf(f, "%63[^:]: %ld %15s\n", key, &value, unit) == 3) {
        if (strcmp(key, "MemTotal") == 0) mem_total = value;
        else if (strcmp(key, "MemAvailable") == 0) mem_avail = value;
        else if (strcmp(key, "SwapTotal") == 0) swap_total = value;
        else if (strcmp(key, "SwapFree") == 0) swap_free = value;
    }
    fclose(f);
    d->mem_total_mb = mem_total / 1024;
    d->mem_avail_mb = mem_avail / 1024;
    d->mem_used_percent = mem_total ? (double)(mem_total - mem_avail) * 100.0 / (double)mem_total : 0.0;
    d->swap_total_mb = swap_total / 1024;
    d->swap_free_mb = swap_free / 1024;
    d->swap_used_percent = swap_total ? (double)(swap_total - swap_free) * 100.0 / (double)swap_total : 0.0;
}

static double read_temp_c(void) {
    char path[128];
    double max_temp = -1.0;
    for (int i = 0; i < 16; i++) {
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", i);
        long raw = read_long_file(path);
        if (raw <= 0) continue;
        double c = raw > 1000 ? raw / 1000.0 : (double)raw;
        if (c > max_temp && c < 140.0) max_temp = c;
    }
    return max_temp;
}

static long read_first_freq_mhz(const char **paths, int count, long divisor) {
    for (int i = 0; i < count; i++) {
        long v = read_long_file(paths[i]);
        if (v > 0) return v / divisor;
    }
    return -1;
}

static void read_disk(PerfData *d) {
    struct statvfs st;
    const char *path = access("/userdata", F_OK) == 0 ? "/userdata" : "/";
    if (statvfs(path, &st) != 0) return;
    unsigned long long total = (unsigned long long)st.f_blocks * st.f_frsize;
    unsigned long long freeb = (unsigned long long)st.f_bavail * st.f_frsize;
    unsigned long long used = total - freeb;
    d->disk_free_gb = (double)freeb / (1024.0 * 1024.0 * 1024.0);
    d->disk_used_percent = total ? (double)used * 100.0 / (double)total : 0.0;
}

static void sample_perf(PerfData *d, CpuStat *prev) {
    CpuStat cur = {0};
    memset(d, 0, sizeof(*d));
    if (read_cpu(&cur) == 0) {
        d->cpu_percent = calc_cpu(prev, &cur);
        *prev = cur;
    }
    read_mem(d);
    d->temp_c = read_temp_c();
    const char *cpu_paths[] = {
        "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    };
    const char *npu_paths[] = {
        "/sys/class/devfreq/fdab0000.npu/cur_freq",
        "/sys/class/devfreq/fdab0000.npu/devfreq/fdab0000.npu/cur_freq"
    };
    d->cpu_mhz = read_first_freq_mhz(cpu_paths, 2, 1000);
    d->npu_mhz = read_first_freq_mhz(npu_paths, 2, 1000000);
    read_disk(d);
}

static void format_lines(const PerfData *d, char lines[][96], int *count) {
    *count = 0;
    snprintf(lines[(*count)++], 96, "CPU      %5.1f%%   %ld MHz", d->cpu_percent, d->cpu_mhz);
    snprintf(lines[(*count)++], 96, "Memory   %5.1f%%   avail %ld MB", d->mem_used_percent, d->mem_avail_mb);
    snprintf(lines[(*count)++], 96, "Swap     %5.1f%%   free  %ld MB", d->swap_used_percent, d->swap_free_mb);
    if (d->temp_c > 0)
        snprintf(lines[(*count)++], 96, "Temp     %5.1f C", d->temp_c);
    else
        snprintf(lines[(*count)++], 96, "Temp     N/A");
    if (d->npu_mhz > 0)
        snprintf(lines[(*count)++], 96, "NPU      %ld MHz", d->npu_mhz);
    else
        snprintf(lines[(*count)++], 96, "NPU      N/A");
    snprintf(lines[(*count)++], 96, "Disk     %5.1f%%   free %.1f GB", d->disk_used_percent, d->disk_free_gb);
}

static void set_window_above(Display *dpy, Window win) {
    Atom wm_state = XInternAtom(dpy, "_NET_WM_STATE", False);
    Atom above = XInternAtom(dpy, "_NET_WM_STATE_ABOVE", False);
    XChangeProperty(dpy, win, wm_state, XA_ATOM, 32, PropModeReplace, (unsigned char *)&above, 1);
}

static void draw_window(Display *dpy, Window win, GC gc, XFontStruct *font, const PerfData *data) {
    char lines[8][96];
    int count = 0;
    format_lines(data, lines, &count);

    XSetForeground(dpy, gc, 0x202020);
    XFillRectangle(dpy, win, gc, 0, 0, 320, 184);
    XSetForeground(dpy, gc, 0x00ff99);
    XDrawString(dpy, win, gc, 12, 24, "RK3588 Board Monitor", 20);
    XSetForeground(dpy, gc, 0xe8e8e8);
    for (int i = 0; i < count; i++) {
        XDrawString(dpy, win, gc, 12, 52 + i * 20, lines[i], (int)strlen(lines[i]));
    }
    XSetForeground(dpy, gc, 0x777777);
    XDrawString(dpy, win, gc, 12, 174, "Drag to move | q/Esc hide", 25);
    (void)font;
}

static int run_window(void) {
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "Cannot open X display. Run inside desktop/NoMachine terminal.\n");
        return 1;
    }
    int screen = DefaultScreen(dpy);
    int sw = DisplayWidth(dpy, screen);
    int x = sw > 340 ? sw - 340 : 20;
    int y = 30;
    Window root = RootWindow(dpy, screen);
    Window win = XCreateSimpleWindow(dpy, root, x, y, 320, 184, 1, 0x444444, 0x202020);
    XStoreName(dpy, win, "Board Monitor");
    set_window_above(dpy, win);
    XSelectInput(dpy, win, ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
    XMapWindow(dpy, win);

    GC gc = XCreateGC(dpy, win, 0, NULL);
    XFontStruct *font = XLoadQueryFont(dpy, "fixed");
    if (font) XSetFont(dpy, gc, font->fid);

    CpuStat prev = {0};
    read_cpu(&prev);
    PerfData data;
    int dragging = 0, drag_x = 0, drag_y = 0;
    time_t last = 0;

    while (1) {
        while (XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == Expose) {
                draw_window(dpy, win, gc, font, &data);
            } else if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_q || ks == XK_Escape) goto done;
            } else if (ev.type == ButtonPress && ev.xbutton.button == Button1) {
                dragging = 1;
                drag_x = ev.xbutton.x_root - x;
                drag_y = ev.xbutton.y_root - y;
            } else if (ev.type == ButtonRelease) {
                dragging = 0;
            } else if (ev.type == MotionNotify && dragging) {
                x = ev.xmotion.x_root - drag_x;
                y = ev.xmotion.y_root - drag_y;
                XMoveWindow(dpy, win, x, y);
            }
        }
        time_t now = time(NULL);
        if (now != last) {
            sample_perf(&data, &prev);
            draw_window(dpy, win, gc, font, &data);
            XFlush(dpy);
            last = now;
        }
        usleep(50000);
    }

done:
    if (font) XFreeFont(dpy, font);
    XFreeGC(dpy, gc);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}

int main(int argc, char **argv) {
    CpuStat prev = {0};
    read_cpu(&prev);
    usleep(200000);
    PerfData data;
    sample_perf(&data, &prev);

    if (argc > 1 && strcmp(argv[1], "--once") == 0) {
        char lines[8][96];
        int count = 0;
        format_lines(&data, lines, &count);
        for (int i = 0; i < count; i++) puts(lines[i]);
        return 0;
    }
    return run_window();
}
