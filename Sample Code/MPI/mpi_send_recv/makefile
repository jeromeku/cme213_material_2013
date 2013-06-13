EXECS=send_recv
MPICC=mpicc

all: ${EXECS}

send_recv: send_recv.c
	${MPICC} -o send_recv send_recv.c

clean:
	rm ${EXECS}
