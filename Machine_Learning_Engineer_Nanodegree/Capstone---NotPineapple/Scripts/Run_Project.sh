### Reset BASH time counter
SECONDS=0

### Delete all output files

echo
echo [INFO] Deleting all output files..

bash Scripts/delete_output_files.sh

echo 

### LeNet Architecture

#### Lenet-25

echo 
echo ====================================================
echo [INFO] Begin LeNet-25...
echo ====================================================
echo 

bash Scripts/LeNet-25.sh

echo 
echo ====================================================
echo [INFO] End LeNet-25...
echo ====================================================
echo [INFO] Completed [1/9]
echo ====================================================
echo 

#### LeNet-50

echo 
echo ====================================================
echo [INFO] Begin LeNet-50...
echo ====================================================
echo

bash Scripts/LeNet-50.sh

echo 
echo ====================================================
echo [INFO] End LeNet-50...
echo ====================================================
echo [INFO] Completed [2/9]
echo ====================================================
echo

#### LeNet-100

echo
echo ====================================================
echo [INFO] Begin LeNet-100...
echo ====================================================
echo

bash Scripts/LeNet-100.sh

echo
echo ====================================================
echo [INFO] End LeNet-100...
echo ====================================================
echo [INFO] Completed [3/9]
echo ====================================================
echo

### Fully Connected Architecture

#### FullConn-25

echo
echo ====================================================
echo [INFO] Begin FullConn-25...
echo ====================================================
echo

bash Scripts/FullConn-25.sh

echo
echo ====================================================
echo [INFO] End FullConn-25...
echo ====================================================
echo [INFO] Completed [4/9]
echo ====================================================
echo

#### FullConn-50

echo
echo ====================================================
echo [INFO] Begin FullConn-50...
echo ====================================================
echo

bash Scripts/FullConn-50.sh

echo
echo ====================================================
echo [INFO] End FullConn-50...
echo ====================================================
echo [INFO] Completed [5/9]
echo ====================================================
echo

#### FullConn-100

echo
echo ====================================================
echo [INFO] Begin FullConn-100...
echo ====================================================
echo

bash Scripts/FullConn-100.sh

echo
echo ====================================================
echo [INFO] End FullConn-100...
echo ====================================================
echo [INFO] Completed [6/9]
echo ====================================================
echo

### Custom CNN Architecture

#### CustomConv-25

echo
echo ====================================================
echo [INFO] Begin CustomConv-25...
echo ====================================================
echo

bash Scripts/CustomConv-25.sh

echo
echo ====================================================
echo [INFO] End CustomConv-25...
echo ====================================================
echo [INFO] Completed [7/9]
echo ====================================================
echo

#### CustomConv-50

echo
echo ====================================================
echo [INFO] Begin CustomConv-50...
echo ====================================================
echo

bash Scripts/CustomConv-50.sh

echo
echo ====================================================
echo [INFO] End CustomConv-50...
echo ====================================================
echo [INFO] Completed [8/9]
echo ====================================================
echo

#### CustomConv-100

echo
echo ====================================================
echo [INFO] Begin CustomConv-100...
echo ====================================================
echo

bash Scripts/CustomConv-100.sh

echo
echo ====================================================
echo [INFO] End CustomConv-100...
echo ====================================================
echo [INFO] Completed [9/9]
echo ====================================================
echo

### Print time elapsed
ELAPSED="Time Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED