### Delete all output files

echo "[INFO] Deleting all output files..."
bash Scripts/delete_output_files.sh

### LeNet Architecture

#### Lenet-25

echo "[INFO] Begin LeNet-25..."
bash Scripts/LeNet-25.sh
echo "[INFO] End LeNet-25..."

#### LeNet-50

echo "[INFO] Begin LeNet-50..."
bash Scripts/LeNet-50.sh
echo "[INFO] End LeNet-50..."

#### LeNet-100

echo "[INFO] Begin LeNet-100..."
bash Scripts/LeNet-100.sh
echo "[INFO] End LeNet-100..."

### Fully Connected Architecture

#### FullConn-25

echo "[INFO] Begin FullConn-25..."
bash Scripts/FullConn-25.sh
echo "[INFO] End FullConn-25..."

#### FullConn-50

echo "[INFO] Begin FullConn-50..."
bash Scripts/FullConn-50.sh
echo "[INFO] End FullConn-50..."

#### FullConn-100

echo "[INFO] Begin FullConn-100..."
bash Scripts/FullConn-100.sh
echo "[INFO] End FullConn-100..."

### Custom CNN Architecture

#### CustomConv-25

echo "[INFO] Begin CustomConv-25..."
bash Scripts/CustomConv-25.sh
echo "[INFO] End CustomConv-25..."

#### CustomConv-50

echo "[INFO] Begin CustomConv-50..."
bash Scripts/CustomConv-50.sh
echo "[INFO] End CustomConv-50..."

#### CustomConv-100

echo "[INFO] Begin CustomConv-100..."
bash Scripts/CustomConv-100.sh
echo "[INFO] End CustomConv-100..."