## 🟨 The purpose of each file

- `all.zip`  includes synthetic charts from [Beagle dataset](https://homes.cs.washington.edu/~leibatt/beagle.html) and real-world cases from the Internet, with a total of 18 chart types and 33,260 images.
- `annotation_18_30k.csv` includes the file name and four primary attributes for each images.
- `all_ftr.npy` includes extracted global features with the shape of (33260, 512).
- `all_ftr_color_hist.npy` includes color histogram information with the shape of (33260, 384).
- `all_ftr_gray.npy` includes trend information with the shape of (33260, 512).