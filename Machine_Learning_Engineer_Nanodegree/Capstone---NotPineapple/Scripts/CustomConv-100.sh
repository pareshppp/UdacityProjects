#!bin/bash/

# Commands to run the scripts

## train model

### CustomConv-100
python Scripts/train_model.py --dataset Input_Files/Training_Images --model NotPineapple --resize 100 --architecture CustomConv --object Pineapple  






## test model - single image

### CustomConv-100
python Scripts/test_model_single_image.py --model Output_Files/Save_Model/NotPineapple_CustomConv_100x100.model --image Input_Files/Testing_Images/Pineapple/Pineapple-3_100.jpg --resize 100 --object Pineapple  

python Scripts/test_model_single_image.py --model Output_Files/Save_Model/NotPineapple_CustomConv_100x100.model --image Input_Files/Testing_Images/NotPineapple/Grape_White_2-r_61_100.jpg --resize 100 --object Pineapple  

# - change the --image based on test images in Testing_Images/ folder






## test model

### CustomConv-100
python Scripts/test_model.py --model Output_Files/Save_Model/NotPineapple_CustomConv_100x100.model --dataset Input_Files/Testing_Images --resize 100 --object Pineapple  

# - change the --object Pineapple with any object and corresponding --model name  