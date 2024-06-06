import albumentations as A

# albumentations transforms for text augmentation
aug_transforms = A.Compose([
    
    # geometric augmentation
    A.Affine(rotate=(-1, 1), shear={'x':(-30, 30), 'y' : (-5, 5)}, scale=(0.6, 1.2), translate_percent=0.02, mode=1, p=0.5),

    # perspective transform
    #A.Perspective(scale=(0.05, 0.1), p=0.5),

    # distortions
    A.OneOf([
        A.GridDistortion(distort_limit=(-.1, .1), p=0.5),
        A.ElasticTransform(alpha=60, sigma=20, alpha_affine=0.5, p=0.5),
    ], p=0.5),

    # erosion & dilation
    A.OneOf([
        A.Morphological(p=0.5, scale=3, operation='dilation'),
        A.Morphological(p=0.5, scale=3, operation='erosion'),
    ], p=0.5),

    # color invertion - negative
    #A.InvertImg(p=0.5),

    # color augmentation - only grayscale images
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    
    # color contrast
    A.RandomGamma(p=0.5, gamma_limit=(80, 120)),
])
