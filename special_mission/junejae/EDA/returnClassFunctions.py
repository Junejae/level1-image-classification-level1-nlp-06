def return_class(row, mask):
    # wear = 0
    # incorrect = 6
    # not_wear = 12
    masked = [0, 6, 12]
    # male = 0
    # female = 3
    gender = [0, 3]
    # under30 = 0
    # btw30and60 = 1
    # over60 = 2
    age = [0, 1, 2]
    # Assuming the mask is labeled as 0,1,2
    # Each of them is 'wear', 'incorrect' and 'not wear'
    mask_index = mask
    gender_index = 0
    age_index = 0
    # Detect Gender
    if row["gender"] == "female":
        gender_index = 1
    # Detect Age
    if 30 <= row["age"] < 60:
        age_index = 1
    elif row["age"] >= 60:
        age_index = 2
    # Print the class number
    return masked[mask_index] + gender[gender_index] + age[age_index]


def return_class_simple(row, mask):
    # Assuming the mask is already labeled as 0,1,2
    # Each of them is 'wear', 'incorrect' and 'not wear'
    gender = 0 if row["gender"] == "male" else 3
    age = min(2, row["age"] // 30)
    # Print the class number
    return mask*6 + gender + age
