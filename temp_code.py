import astrolib as al

def sum_library_objects(lo_array):
    sum_data = np.zeros(( len(lo_array[0]), len(lo_array[0][0])))
    for object in lo_array:
        for i in range(len(object)):
            sum_data[i][0] = object[i][0]
            sum_data[i][1] += object[i][1]

    return sum_data



def calc_chi2_err(obs, obs_err, library_array):
    err = 0

    for i in range(len(obs)):
        lib_sum = 0
        for lib in library_array:
            lib_sum += lib[i]

        err += ((obs[i] - lib_sum) / obs_err[i])**2

    return(err)







def sum_astro_data(m1, m2, index = 'lambda_em', value = 1):
    i1 = 0
    i2 = 0
    max_len = max(len(m1), len(m2))
    sum_index = []
    sum_value = []

    for i in range(0, max_len):
        # print(f"Round {i}, index 1: {i1}, index 2: {i2}.")
        
        if i1 >= len(m1[index]):
            # print("First index already ended.")
            sum_index.append( m2[index][i2])
            sum_value.append( m2[value][i2] )
            i2 += 1
        elif i2 >= len(m2[index]):
            # print("Second index already ended.")
            sum_index.append( m1[index][i1])
            sum_value.append( m1[value][i1] )
            i1 += 1
        elif m1[index][i1] == m2[index][i2]:
            sum_index.append( m1[value][i1])
            sum_value.append( m1[value][i1] + m2[value][i2])
            i1 += 1
            i2 += 1
        elif m1[index][i1] > m2[index][i2]:
            # print("First index ahead of second.")
            sum_index.append( m2[index][i2])
            sum_value.append( m2[value][i2] )
            i2 += 1
        else:
            # print("Second index ahead of first.")
            sum_index.append( m1[index][i1])
            sum_value.append( m1[value][i1] )
            i1 += 1

    df = pd.DataFrame()
    df[index] = sum_index
    df[value] = sum_value

    return LibraryObject('sum', df )

