EXECS=ring_DL ring_NS ring_SEC ring_SR
MPICC=mpicc

all: ${EXECS}

ring_DL: ring_DL.c
	${MPICC} -o $@ $<

ring_NS: ring_NS.c
	${MPICC} -o $@ $<

ring_SEC: ring_SEC.c
	${MPICC} -o $@ $<

ring_SR: ring_SR.c
	${MPICC} -o $@ $<

clean:
	rm ${EXECS}
