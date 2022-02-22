import pandas as pd

# inspired by 이승현_T3155
def preprocessFunction():
    train_dir = "/opt/ml/input/data/train"

    file_names = [
        "/incorrect_mask.jpg",
        "/mask1.jpg",
        "/mask2.jpg",
        "/mask3.jpg",
        "/mask4.jpg",
        "/mask5.jpg",
        "/normal.jpg",
    ]

    prepro_data_info = pd.DataFrame(columns={"id", "img_path", "race", "mask", "gender", "age", "label"})

    all_id, all_path, all_race, all_mask, all_age, all_gender, all_label = ([],[],[],[],[],[],[])

    data_info = pd.read_csv(train_dir + "/train.csv", header=0)

    for i in range(len(data_info.id)):
        for f_name in file_names:
            mask = 0
            if "incorrect" in f_name:
                mask = 1
            elif "normal" in f_name:
                mask = 2

            gender = gender = 0 if data_info.iloc[i].gender == "male" else 1

            age = min(2, data_info.iloc[i].age // 30)

            all_id.append(data_info.iloc[i].id)
            all_path.append(data_info.iloc[i].path + f_name)
            all_race.append(data_info.iloc[i].race)
            all_mask.append(mask)
            all_gender.append(gender)
            all_age.append(age)
            all_label.append(mask * 6 + gender * 3 + age)

    prepro_data_info["id"] = all_id
    prepro_data_info["img_path"] = all_path
    prepro_data_info["race"] = all_race
    prepro_data_info["mask"] = all_mask
    prepro_data_info["gender"] = all_gender
    prepro_data_info["age"] = all_age
    prepro_data_info["label"] = all_label

    return prepro_data_info
