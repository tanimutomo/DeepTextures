
def check_array(x, name):
    print("-" * 10, name, "-" * 10)
    print("dtype: {} / "
          "shape: {} / "
          "min: {:.4f} / "
          "mean: {:.4f} / "
          "max: {:.4f}".format(
              x.dtype, x.shape, x.min(), x.mean(), x.max()
              )
          )
    print("-" * 10 + "-" * (len(name)+2) + "-" * 10)
