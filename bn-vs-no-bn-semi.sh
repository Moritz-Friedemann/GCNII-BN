echo "2 layers no batchnorm"
python -u train.py --data cora --layer 2
echo "4 layers no batchnorm"
python -u train.py --data cora --layer 4
echo "8 layers no batchnorm"
python -u train.py --data cora --layer 8
echo "16 layers no batchnorm"
python -u train.py --data cora --layer 16
echo "32 layers no batchnorm"
python -u train.py --data cora --layer 32
echo "64 layers no batchnorm"
python -u train.py --data cora --layer 64

echo "2 layers no batchnorm"
python -u train.py --data cora --layer 2 --batchnorm
echo "4 layers no batchnorm"
python -u train.py --data cora --layer 4 --batchnorm
echo "8 layers no batchnorm"
python -u train.py --data cora --layer 8 --batchnorm
echo "16 layers no batchnorm"
python -u train.py --data cora --layer 16 --batchnorm
echo "32 layers no batchnorm"
python -u train.py --data cora --layer 32 --batchnorm
echo "64 layers no batchnorm"
python -u train.py --data cora --layer 64 --batchnorm