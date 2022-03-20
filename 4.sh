
for i in {1..10..1}
    do 
        export OMP_NUM_THREADS=$i
        echo "Number of threads $i"
        for n in {50..300..50}
        do
            echo "N = $n"
            ./jacobi2D-omp $n
            ./gs2D-omp $n
        done
    done