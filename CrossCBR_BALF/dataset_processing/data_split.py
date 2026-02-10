import random
from collections import defaultdict

# 读取数据文件
def load_data(file_path):
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            user_id, bundle_id = line.strip().split()
            interactions.append((user_id, bundle_id))
    return interactions

# 读取数据
train_data = load_data('../datasets/Youshu_IID/user_bundle_train.txt')
tune_data = load_data('../datasets/Youshu_IID/user_bundle_tune.txt')
test_data = load_data('../datasets/Youshu_IID/user_bundle_test.txt')

all_data = train_data + tune_data + test_data

# train:tune:test = 7:1:2
total_interactions = len(all_data)
test_size = total_interactions // 5  # 20%
tune_size = total_interactions // 10  # 10%
print(f'Total interaction size: {total_interactions}')
print(f'Expected Test data size: {test_size}')
print(f'Expected Tune data size: {tune_size}')
print(f'Expected Train data size: {total_interactions - test_size - tune_size}')

# 按 bundle_id 分组
bundle_dict = defaultdict(list)
for user_id, bundle_id in all_data:
    bundle_dict[bundle_id].append((user_id, bundle_id))

# 采样测试集
def sample_data(bundle_dict, min_samples, sample_size, task):
    sampled_data = []
    remaining_data = []

    all_bundles = list(bundle_dict.items())
    random.shuffle(all_bundles)  # 打乱 bundle 处理顺序，防止数据偏差

    if task == 'tune':
        threshold = 12
    elif task == 'test':
        threshold = 12

    for bundle_id, interactions in all_bundles:
        if len(interactions) >= threshold:
            selected_samples = random.sample(interactions, min_samples)
            sampled_data.extend(selected_samples)
            remaining_data.extend([item for item in interactions if item not in selected_samples])
        else:
            remaining_data.extend(interactions)
    print(f"len(sampled_{task}_data): {len(sampled_data)}")

    # 如果数据不足，从剩余数据中补充
    if len(sampled_data) < sample_size:
        extra_samples = random.sample(remaining_data, sample_size - len(sampled_data))
        sampled_data.extend(extra_samples)
        extra_set = set(extra_samples)
        remaining_data = [item for item in remaining_data if item not in extra_set]

    return sampled_data, remaining_data

# test data
test_data_final, remaining_data = sample_data(bundle_dict, min_samples=7, sample_size=test_size, task='test')

bundle_dict_tune = defaultdict(list)
for user_id, bundle_id in remaining_data:
    bundle_dict_tune[bundle_id].append((user_id, bundle_id))

# tune data
tune_data_final, remaining_data = sample_data(bundle_dict_tune, min_samples=7, sample_size=tune_size, task='tune')
# tune_data_final = random.sample(remaining_data, tune_size)

# train data
train_data_final = remaining_data
# train_data_final = [item for item in remaining_data if item not in tune_data_final]

# 输出数据大小
print(f'Total interaction size: {total_interactions}')
print(f'Final test data size: {len(test_data_final)} (Expected: {test_size})')
print(f'Final tune data size: {len(tune_data_final)} (Expected: {tune_size})')
print(f'Train data size: {len(train_data_final)} (Expected: {total_interactions - test_size - tune_size})')

# 保存数据集
def save_data(file_name, data):
    with open(file_name, 'w') as f:
        for user_id, bundle_id in data:
            f.write(f"{user_id}\t{bundle_id}\n")

# OOD 数据集
save_data('../datasets/Youshu/user_bundle_train.txt', train_data_final)
save_data('../datasets/Youshu/user_bundle_tune.txt', tune_data_final)
save_data('../datasets/Youshu/user_bundle_test.txt', test_data_final)

print("Completed dataset splitting!")
