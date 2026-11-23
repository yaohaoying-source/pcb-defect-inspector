CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?= -lX11

all: board_monitor

board_monitor: board_monitor.c
	$(CC) $(CFLAGS) board_monitor.c -o board_monitor $(LDFLAGS)

clean:
	rm -f board_monitor
