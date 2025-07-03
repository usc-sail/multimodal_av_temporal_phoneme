from torch.utils.data import DataLoader
from dataloader import VideoAudioPhonemeDataset

def get_dataset(video_directory, batch_size):
    train_set = VideoAudioPhonemeDataset(video_directory, training=True)
    test_set = VideoAudioPhonemeDataset(video_directory, training=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader