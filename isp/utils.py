import sys

def delete_multiple_lines(n=1):
    """
    Delete the last line in the STDOUT.
    From https://stackoverflow.com/questions/19596750/is-there-a-way-to-clear-your-printed-text-in-python
    """
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line
