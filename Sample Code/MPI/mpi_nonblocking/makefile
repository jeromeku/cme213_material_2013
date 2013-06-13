EXECS=gather_ring
MPICC=mpic++

all: ${EXECS}

gather_ring: gather_ring.C
	${MPICC} -o $@ $<

clean:
	rm ${EXECS}
