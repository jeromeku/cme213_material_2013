all: create_cipher solve_cipher

create_cipher: create_cipher.cu
	nvcc -o create_cipher create_cipher.cu -O3 -arch=sm_20

solve_cipher: solve_cipher.cu
	nvcc -o solve_cipher solve_cipher.cu -O3 -arch=sm_20

clean:
	rm -f create_cipher solve_cipher rm cipher_text.txt plain_text.txt
