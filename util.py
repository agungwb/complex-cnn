import numpy as np

def save_matrix(data, filename, delimiter, mode = '2d'):
    if mode == '1d':
        if len(data.shape) == 1:
            np.savetxt(filename, data, delimiter=delimiter)
        elif len(data.shape) == 2:
            with open(filename, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                # outfile.write('# Array shape: {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                outfile.write('# Array shape: {0}\n\n'.format(data.shape))
                i = 0
                for data_slice in data:
                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.
                    outfile.write("({0},:)\n".format(str(i)))
                    np.savetxt(outfile, data_slice, delimiter=delimiter)
                    outfile.write("\n")

                    # Writing out a break to indicate different slices...
                    i = i+1
        elif len(data.shape) == 3:
            with open(filename, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                # outfile.write('# Array shape: {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                outfile.write('# Array shape: {0}\n\n'.format(data.shape))
                i = 0
                for data_slice in data:
                    j = 0
                    for data_slice_2 in data_slice:
                        # The formatting string indicates that I'm writing out
                        # the values in left-justified columns 7 characters in width
                        # with 2 decimal places.
                        outfile.write("({0},{1},:)\n".format(str(i), str(j)))
                        np.savetxt(outfile, data_slice_2, delimiter=delimiter)

                        # Writing out a break to indicate different slices...
                        outfile.write("\n")
                        j = j + 1
                    i = i + 1
        elif len(data.shape) == 4:
            with open(filename, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                # outfile.write('# Array shape: {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                outfile.write('# Array shape: {0}\n\n'.format(data.shape))
                i = 0
                for data_slice in data:
                    j = 0
                    for data_slice_2 in data_slice:
                        k = 0
                        for data_slice_3 in data_slice_2:
                            # The formatting string indicates that I'm writing out
                            # the values in left-justified columns 7 characters in width
                            # with 2 decimal places.
                            outfile.write("({0},{1},{2},:)\n".format(str(i),str(j),str(k)))
                            np.savetxt(outfile, data_slice_3, delimiter=delimiter)

                            # Writing out a break to indicate different slices...
                            outfile.write("\n")
                            k = k + 1
                        j = j + 1
                    i = i + 1
        else:
            print("Not supported : {} failed to create".format(filename))
    elif mode == '2d':
        if len(data.shape) == 1 or len(data.shape) == 2:
            np.savetxt(filename, data, delimiter=delimiter)
        elif len(data.shape) == 3:
            with open(filename, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                # outfile.write('# Array shape: {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                outfile.write('# Array shape: {0}\n\n'.format(data.shape))
                i = 0
                for data_slice in data:
                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.
                    outfile.write("({0},:,:)\n".format(str(i)))
                    np.savetxt(outfile, data_slice, delimiter=delimiter)
                    outfile.write("\n")

                    # Writing out a break to indicate different slices...
                    i = i+1
        elif len(data.shape) == 4:
            with open(filename, 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                # outfile.write('# Array shape: {0}\n'.format(data.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                outfile.write('# Array shape: {0}\n\n'.format(data.shape))
                i = 0
                for data_slice in data:
                    j = 0
                    for data_slice_2 in data_slice:
                        # The formatting string indicates that I'm writing out
                        # the values in left-justified columns 7 characters in width
                        # with 2 decimal places.
                        outfile.write("({0},{1},:,:)\n".format(str(i),str(j)))
                        np.savetxt(outfile, data_slice_2, delimiter=delimiter)

                        # Writing out a break to indicate different slices...
                        outfile.write("\n")
                        j = j + 1
                    i = i + 1
        else:
            print("Not supported : {} failed to create".format(filename))
    else:
        print("Not supported : {} failed to create".format(filename))

def normalize(a):
    max_magnitude = np.max(np.abs(a))
    a_normalized = a / max_magnitude
    return a_normalized

def normalize_OLD(a):
    max = np.max(a)
    min = np.min(a)
    a_normalized = (a-min)/(max-min)
    return a_normalized