# Linux commands to extract NotPineapple dataset from Kaggle-fruits-360 dataset


## Training Images

### commands run from Capstone--Object-NotObject/fruits-360/Training directory
cd fruits-360/Training

#### add directory name to files
rnm -ns '/pd0/-/fn/' */*.jpg

#### replace space with underscore in directory and file names
rnm -rs '/ /_/g' ./* -dp -1

#### move all files from subfolders to main folder
find . -mindepth 2 -type f -print -exec mv {} . \;

#### delete all empty directories
find . -type d -empty -delete

#### create directories
mkdir Pineapple NotPineapple NotPineapple_All

#### move all jpg files to subdirectory
mv *.jpg NotPineapple_All

#### move all Pineapple files to Pineapple directory
mv NotPineapple_All/*Pineapple* Pineapple/

#### randomly select 490 files from NotPineapple_All and move to NotPineapple
shuf -n 490 -e NotPineapple_All/* | xargs -i mv {} NotPineapple/



### commands run from Capstone--Object-NotObject directory
cd ..; cd ..

#### copy Pineapple and NotPineapple directories to Capstone--Object-NotObject/Training_Images directory
cp -r fruits-360/Training/Pineapple Training_Images/
cp -r fruits-360/Training/NotPineapple Training_Images/








## Testing Images

### commands run from Capstone--Object-NotObject/fruits-360/Validation directory
cd fruits-360/Validation

#### add directory name to files
rnm -ns '/pd0/-/fn/' */*.jpg

#### replace space with underscore in directory and file names
rnm -rs '/ /_/g' ./* -dp -1

#### move all files from subfolders to main folder
find . -mindepth 2 -type f -print -exec mv {} . \;

#### delete all empty directories
find . -type d -empty -delete

#### create directories
mkdir Pineapple NotPineapple NotPineapple_All

#### move all jpg files to subdirectory
mv *.jpg NotPineapple_All

#### move all Pineapple files to Pineapple directory
mv NotPineapple_All/*Pineapple* Pineapple/

#### randomly select 490 files from NotPineapple_All and move to NotPineapple
shuf -n 166 -e NotPineapple_All/* | xargs -i mv {} NotPineapple/



### commands run from Capstone--Object-NotObject directory
cd ..; cd ..

#### copy Pineapple and NotPineapple directories to Capstone--Object-NotObject/Training_Images directory
cp -r fruits-360/Validation/Pineapple Testing_Images/
cp -r fruits-360/Validation/NotPineapple Testing_Images/


s
