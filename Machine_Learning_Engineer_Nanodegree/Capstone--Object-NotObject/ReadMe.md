## train model
python train_model.py --dataset Training_Images --model Pineapple-NotPineapple --resize 28 --architecture LeNet --object Pineapple

## test model - single image
python test_model_single_image.py --model Pineapple-NotPineapple_LeNet_28x28.model --image Testing_Images/Pineapple/Pineapple-3_100.jpg --resize 28 --object Pineapple
python test_model_single_image.py --model Pineapple-NotPineapple_LeNet_28x28.model --image Testing_Images/NotPineapple/Grape_White_2-r_61_100.jpg --resize 28 --object Pineapple

 - change the --image based on test images in Testing_Images/ folder

## test model
python test_model.py --model Pineapple-NotPineapple_LeNet_28x28.model --dataset Testing_Images --resize 28 --object Pineapple

 - change the --object Pineapple with any object and corresponding --model name