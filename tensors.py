from edtrace import text, link, image, note
import numpy as np
import torch
from einops import einsum, reduce, rearrange
import timeit


def main():
    text("# Tensors")
    image("images/tensor.jpg", width=200), note("This is probably what you think of when you hear the word \"tensor\".")
    text("**Tensors** are the atoms of modern machine learning.")
    text("They are used to represent data, model parameters, gradients, intermediate computations (activations), etc.")
    text("Tensors show up elsewhere in science and engineering, so they're generally useful to learn.")
    text("We'll introduce the core ideas through NumPy examples.")

    creating_tensors()
    tensor_examples()
    viewing_tensors()
    elementwise_operations()
    matrix_multiplication()
    efficiency()
    einops()

    text("Summary")
    text("- Use tensors to represent everything")
    text("- Express computations using a few tensor operations for efficiency (a puzzle!)")
    text("- Use einops to make computations more legible")
    text("- Practice (it's like learning a new language)!")


def creating_tensors():
    text("Tensors are represented by multi-dimensional arrays (generalizing vectors and matrices).")

    text("The simplest tensor is simply a **scalar** (rank 0 tensor).")
    x = np.array(42)  # @inspect x
    s = x.shape  # @inspect s

    text("A **vector** is a rank 1 tensor.")  # @clear x s
    x = np.array([1, 2, 3])  # @inspect x
    s = x.shape  # @inspect s

    text("A **matrix** is a rank 2 tensor.")  # @clear x s
    x = np.array([  # @inspect x
        [1, 2, 3],
        [4, 5, 6],
    ])
    s = x.shape  # @inspect s

    text("In general, a **tensor** can be any rank.  Here's a rank 3 tensor.")  # @clear x s
    x = np.array([  # @inspect x
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
    ])
    s = x.shape  # @inspect s

    text("You can extract slices from this tensor:")  # @clear s
    y = x[1]  # @inspect y
    y = x[1][0]  # @inspect y
    y = x[1][0][2]  # @inspect y

    text("You usually don't create tensors by writing down all the entries.")  # @clear x y

    text("You can use special commands to create structured tensors.")

    # Create a 2x3 array of zeros
    x = np.zeros((2, 3))  # @inspect x

    # Create a 2x3 array of ones
    x = np.ones((2, 3))  # @inspect x

    # Create a 2x3 array filled with random numbers from a normal distribution
    x = np.random.randn(2, 3)  # @inspect x

    # Create a 3x3 identity matrix
    x = np.eye(3)  # @inspect x

    # Create a matrix where the diagonal is a vector
    x = np.diag([1, 2, 3])  # @inspect x

    text("Or you can read / write them from disk")
    np.save("var/x.npy", x)
    new_x = np.load("var/x.npy")  # @inspect new_x


def tensor_examples():
    text("Here are some typical tensors that show up in machine learning.")

    # A D-dimensional data point
    D = 2  # Number of dimensions
    x = np.ones(D)  # @inspect x
    text("Here, `np.ones` is arbitrary; we're just trying to get a tensor of the right shape.")

    text("We often batch examples together (for efficiency).")  # @clear x
    N = 3  # Number of examples
    # A dataset of N examples, each D-dimensional point
    x = np.ones((N, D))  # @inspect x

    text("In language modeling, each example is a whole sequence of length.")  # @clear x
    L = 4  # Length of sequence
    # A dataset of N examples, each length L, each position is D-dimensional
    x = np.ones((N, L, D))  # @inspect x

    text("In vision, images have")  # @clear x
    H = 2  # A height
    W = 2  # A width
    C = 3  # A number of channels (red, green, blue)
    x = np.ones((N, H, W, C))

    text("In a neural network, we have a weight matrix that transforms a")  # @clear x
    Din = 3  # ...a Din-dimensional input
    Dout = 2  # ...to a Dout-dimensional output
    w = np.ones((Din, Dout))  # @inspect w

    text("In general, the parameters of a neural network model are a collection of tensors.")
    text("For example, here are the parameters of the DeepSeek v3 model "), link("https://arxiv.org/abs/2412.19437")
    link("https://huggingface.co/deepseek-ai/DeepSeek-V3?show_file_info=model.safetensors.index.json", title="DeepSeek-V3 on Hugging Face")


def viewing_tensors():
    text("Given a tensor, you can extract parts of it.")
    x = np.random.randn(2, 3)  # @inspect x

    # Get row 0
    y = x[0]  # @inspect y

    # Get column 1
    y = x[:, 1]  # @inspect y

    # Transpose the matrix
    y = x.transpose(1, 0)  # @inspect y

    text("Note that these operations do not make a copy of the tensor.")
    text("...so if you modify the tensor, you modify the view.")
    text("Check that mutating x also mutates y.")
    x[0][0] = 100  # @inspect x y
    text("Generally, be careful and don't mutate tensors if you don't need to!")


def elementwise_operations():
    text("These operations apply some operation to each element of the tensor")
    text("...and return a (new) tensor of the same shape.")

    x = np.array([1, 4, 9])  # @inspect x
    y = np.power(x, 2)  # @inspect y
    y = np.sqrt(x)  # @inspect y

    y = x + x  # @inspect y
    y = x * 3  # @inspect y
    y = x / 2  # @inspect y

    text("You can create zeros and ones with the same shape as another tensor:")
    y = np.zeros_like(x)  # @inspect y
    y = np.ones_like(x)  # @inspect y


def matrix_multiplication():
    text("Finally, the bread and butter of deep learning: matrix multiplication.")
    x = np.ones((4, 6))  # @inspect x
    w = np.ones((6, 3))  # @inspect w
    y = x @ w  # @inspect y
    assert y.shape == (4, 3)

    text("We can batch multiple matrix operations:")  # @clear y
    x = np.ones((2, 4, 6))  # @inspect x
    w = np.ones((6, 3))  # @inspect w
    y = x @ w  # @inspect y
    assert y.shape == (2, 4, 3)
    text("In this case, for each slice x[0], x[1], ..., we multiply by `w`.")
    text("Terminology: w is **broadcasted** to each slice of x.")


def efficiency():
    text("There are often multiple ways to compute the same result.")
    text("Whenever possible, try to express computation as tensor operations.")
    N = 16
    x = np.ones((N, N))
    w = np.ones((N, N))

    text("Let's compute a matrix multiplication in Python.")
    def slow_matmul() -> np.ndarray:
        y = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    y[i, j] += x[i, k] * w[k, j]
        return y
    python_time = timeit.timeit(slow_matmul, number=1)  # @inspect python_time @stepover

    text("Let's now do it in NumPy.")
    numpy_time = timeit.timeit(lambda: x @ w, number=1)  # @inspect numpy_time @stepover

    # How much faster is NumPy than Python?
    speedup = python_time / numpy_time  # @inspect speedup

    text("Matrix multiplication is even faster on GPUs (for large matrices)!")

def einops():
    einops_motivation()

    text("Einops is a library for manipulating tensors where dimensions are named.")
    text("It is inspired by Einstein summation notation (Einstein, 1916).")
    link(title="[Einops tutorial]", url="https://einops.rocks/1-einops-basics/")

    einops_einsum()
    einops_reduce()
    einops_rearrange()


def einops_motivation():
    text("Traditional PyTorch code:")
    x = torch.ones(2, 2, 3)      # batch seq hidden  @inspect x
    y = torch.ones(2, 2, 3)      # batch seq hidden  @inspect y
    z = x @ y.transpose(-2, -1)  # batch seq seq  @inspect z
    text("Easy to mess up the dimensions (what is -2, -1?)...")


def einops_einsum():
    text("Einsum is generalized matrix multiplication with good bookkeeping.")

    x = torch.ones(3, 4)  # seq1 hidden @inspect x
    y = torch.ones(4, 3)  # hidden seq2 @inspect y

    # Old way
    z = x @ y   # seq1 seq2  @inspect z

    # New (einops) way
    z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # @inspect z

    text("Let's try a more complex example...")  # @clear x y z

    x = torch.ones(2, 3, 4)  # batch seq1 hidden @inspect x
    y = torch.ones(2, 3, 4)  # batch seq2 hidden @inspect y

    # Old way
    z = x @ y.transpose(-2, -1)  # batch seq1 seq2  @inspect z

    # New (einops) way
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    text("Dimensions that are not named in the output are summed over.")

    text("We can use `...` to represent broadcasting over any number of dimensions (instead of writing `batch`):")
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z


def einops_reduce():
    text("You can reduce a single tensor via some operation (e.g., sum, mean, max, min).")
    x = torch.ones(2, 3, 4)  # batch seq hidden @inspect x

    # Old way
    y = x.sum(dim=-1)  # @inspect y

    # New (einops) way
    y = reduce(x, "... hidden -> ...", "sum")  # @inspect y


def einops_rearrange():
    text("Sometimes, a dimension represents two dimensions")
    text("...and you want to operate on one of them.")

    x = torch.ones(3, 8)  # seq total_hidden @inspect x
    text("...where `total_hidden` is a flattened representation of `heads * hidden1`")
    w = torch.ones(4, 4)  # hidden1 hidden2 @inspect w

    text("Break up `total_hidden` into two dimensions (`heads` and `hidden1`):")
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # @inspect x

    text("Perform the transformation by `w`:")
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # @inspect x

    text("Combine `heads` and `hidden2` back together:")
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # @inspect x


if __name__ == "__main__":
    main()
