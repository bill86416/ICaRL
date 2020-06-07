for t in 1 2 3
do
for c in 2 5 10
do
python main.py --trial $t --dataset mnist --num_classes $c
done
done
