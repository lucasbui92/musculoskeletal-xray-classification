from torchvision import transforms

def get_transforms():
    # Standardize to 256x256 then crop to 224x224
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(p=0.5),      
        transforms.RandomRotation(15),               
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_test
