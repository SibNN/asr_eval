# Function locate_subarray_in_array (defined in asr_eval/utils/misc.py at lines 73-85)

def locate_subarray_in_array[T: (INTS, FLOATS)](
    arr: T, subarr: T
) -> list[int]:
    """Finds all positions X where :code:`arr[X:X+len(subarr)]` equals
    :code:`subarr`, in effiecient way.
    """
    ...