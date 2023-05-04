import os

if __name__ == "__main__":
    set1 = os.listdir("landscape/train")
    set2 = os.listdir("svhn/train")

    print(len(set1))
    print(len(set2))

    """for i in range(len(set1), len(set2)):
        os.remove(f"svhn/train/{set2[i]}")"""