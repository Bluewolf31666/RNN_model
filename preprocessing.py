from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import datasets, transforms
import torch
import setting 
class data():
    def VGGnet():
        training_path = setting.training_path
        testing_path = setting.test_path
        targetSize = (150, 150)
        Generator = ImageDataGenerator(rescale = 1./255,
                                    validation_split = 0.15)

        target_labels = ["glioma", "meningioma", "notumor", "pituitary"]
        train_gen = Generator.flow_from_directory(training_path,
                                                # resizing training images to (min(heights, widths), min(heights, widths)) in both training and testing sets
                                                target_size = targetSize,
                                                color_mode = 'rgb',
                                                classes = target_labels,
                                                class_mode = 'categorical',
                                                batch_size = 32,
                                                shuffle = True,
                                                subset = 'training')

        val_gen = Generator.flow_from_directory(training_path,
                                                # resizing validation images to (min(heights, widths), min(heights, widths)) in both training and testing sets
                                                target_size = targetSize,
                                                color_mode = 'rgb',
                                                classes = target_labels,
                                                class_mode = 'categorical',
                                                batch_size = 32,
                                                shuffle = True,
                                                subset = 'validation')

        print("Training batch classes: ", train_gen.class_indices)
        print("Validation batch classes: ", val_gen.class_indices)
        
        return train_gen,val_gen
    def Alex():
        data_dir= 'D:/archive'
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(((224,224))),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train_data=datasets.ImageFolder(data_dir+'/Training',transform=transform)
        test_data=datasets.ImageFolder(data_dir+'/Testing',transform=transform)
        train_dataloader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
        test_dataloader=torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)
        return train_dataloader,test_dataloader