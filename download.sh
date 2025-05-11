mkdir data
mkdir temp
curl "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -o temp/ann.zip
curl "http://images.cocodataset.org/zips/train2014.zip" -o temp/img.zip
unzip temp/ann.zip -d data
unzip temp/img.zip -d data
