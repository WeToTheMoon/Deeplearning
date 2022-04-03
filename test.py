import dictlearn as dl

print(dl.__version__)

x = dl.dct_dict(10, 50)

print(x)

dl.visualize_dictionary(x, 16, 16)

