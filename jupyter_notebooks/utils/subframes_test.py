    def __init__(self, full_img_dir, full_img_anno_path, sf_target_type='coco', sf_transforms=None):
    """
    Instantiate a 'CustomDataset' dataset.

    Parameters
    ----------
    img_dir (str): TODO: Add information.
    img_ann_path (str): TODO: Add information.
    target_type (str): (default: "coco") #TODO: Add information.
    transforms (???): (default: None) #TODO: Add information.
    """
        self.full_img_dir = full_img_dir
        self.full_img_anno_path = full_img_anno_path
        self.sf_target_type = sf_target_type
        self.sf_transforms = sf_transforms

        with open(file = full_img_anno_path, mode = "r") as json_file:
            self.data = json.load(fp = json_file)

    
def subexport(full_img_dir, full_img_anno_path, sf_width, sf_height, sf_output_dir, sf_overlap=False, sf_strict=False, print_rate=50, sf_object_only=True, sf_anno_export=True):
    """
    Slices an image and its associated annotations in subframes that can be exported along with their newly generated annotations. This function uses the 'Subframes' class for image processing.

    Parameters
    ----------
    full_img_dir (str): Path to the directory containing the original unsliced images.
    full_img_anno_path (str): Path to the annotation file in the COCO format (.json) for the original unsliced images.
    sf_width (int): Width of the newly generated sub-frames (i.e. sliced images).
    sf_height (int): Height of the newly generated sub-frames (i.e. sliced images).
    sf_output_dir (str): Path to the directory in which to save the newly generated sub-frames and annotations.
    sf_overlap (bool, optional): If set to 'True', an overlap of 50 % will be considered between two newly generated consecutive sub-frames. (default: False)
    sf_strict (bool, optional): If set to 'True', newly generated subframes will be of the same exact size. (default: False)
    print_rate (int, optinal): Console print rate for the image processing progress. (default: 50)
    sf_object_only (bool, optional): If set to 'True', only sub-frames containing objects will be saved. If set to 'False', all sub-frames will be saved. (default: True)
    sf_anno_export (bool, optional): If set to 'True', newly generated annotations will be exported. If set to 'False', newly generated annotations will not be exported. (default: True)

    Returns
    -------
    return_var (list): A coco-type JSON file named 'coco_subframes.json' is created inside the subframes' folder.
    """
    # Open and load the annotation file for the unsliced images.
    with open(file = full_img_anno_path, mode = "r") as json_file:
        coco_dic = json.load(fp = json_file)

    # Creating the custom dataset using the original unsliced images and their associated annotations.
    dataset = CustomDataset(
        full_img_dir = full_img_dir,
        full_img_anno_path = full_img_anno_path,
        sf_target_type = "coco"
        sf_transforms = None
    )

    # Creating a sampler using PyTorch's 'SequentialSampler' module.
    # A sampler sequentially samples elements for a given dataset, and always in the same order.
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler
    sampler = torch.utils.data.SequentialSampler(
        data_source = dataset
    )

    # Collate_fn
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=1,
                                            sampler=sampler,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    # Header
    all_results = [['filename','boxes','labels','HxW']]

    # intial time
    t_i = time.time()

    for i, (image, target) in enumerate(dataloader):

        if i == 0:
            print(' ')
            print('-'*38)
            print('Sub-frames creation started...')
            print('-'*38)

        elif i == len(dataloader)-1:
            print('-'*38)
            print('Sub-frames creation finished!')
            print('-'*38)

        image = image[0]
        target = target[0]

        # image id and name
        img_id = int(target['image_id'])
        for im in coco_dic['images']:
            if im['id'] == img_id:
                img_name = im['file_name']

        # Get subframes
        sub_frames = Subframes(img_name, image, target, sf_width, sf_height, strict=sf_strict)  #TODO: Verify this line of code.
        results = sub_frames.getlist(overlap=sf_overlap)    #TODO: Verify this line of code.

        # Save
        sub_frames.save(results, output_path=sf_output_dir, object_only=sf_object_only)
        
        if sf_object_only is True:      #TODO: Verify this line of code.
            for b in range(len(results)):
                if results[b][1]:
                    h = np.shape(results[b][0])[0]
                    w = np.shape(results[b][0])[1]
                    all_results.append([results[b][3],results[b][1],results[b][2],[h,w]])

        elif sf_object_only is not True:    #TODO: Verify this line of code.
            for b in range(len(results)):
                h = np.shape(results[b][0])[0]
                w = np.shape(results[b][0])[1]
                all_results.append([results[b][3],results[b][1],results[b][2],[h,w]])

        if i % print_rate == 0:
            print('Image [{:<4}/{:<4}] done.'.format(i, len(coco_dic['images'])))

    # final time
    t_f = time.time()

    print('Elapsed time : {}'.format(str(datetime.timedelta(seconds=int(np.round(t_f-t_i))))))
    print('-'*38)
    print(' ')

    return_var = np.array(all_results)[:,:3].tolist()

    # Export new annos
    if sf_anno_export is True:
        file_name = 'coco_subframes.json'
        output_f = os.path.join(sf_output_dir, file_name)

        # Initializations
        images = []
        annotations = []
        id_img = 0
        id_ann = 0

        for i in range(1,len(all_results)):
            
            id_img += 1

            h = all_results[i][3][0]
            w = all_results[i][3][1]

            dico_img = {
                "license": 1,
                "file_name": all_results[i][0],
                "coco_url": "None",
                "height": h,    #TODO: Verify if 'height' or 'sf_height'
                "width": w,     #TODO: Verify if 'width' or 'sf_width'
                "date_captured": "None",
                "flickr_url": "None",
                "id": id_img
            }

            images.append(dico_img)

            # Bounding boxes
            if all_results[i][1]:
                
                bndboxes = all_results[i][1]

                for b in range(len(bndboxes)):

                    id_ann += 1

                    bndbox = bndboxes[b]
                    
                    # Convert 
                    x_min = int(np.round(bndbox[0]))
                    y_min = int(np.round(bndbox[1]))
                    box_w = int(np.round(bndbox[2]))
                    box_h = int(np.round(bndbox[3]))

                    coco_box = [x_min,y_min,box_w,box_h]

                    # Area
                    area = box_w*box_h

                    # Label
                    label_id = all_results[i][2][b]

                    # Store the values into a dict
                    dico_ann = {
                            "segmentation": [[]],
                            "area": area,
                            "iscrowd": 0,
                            "image_id": id_img,
                            "bbox": coco_box,
                            "category_id": label_id,
                            "id": id_ann
                    }

                    annotations.append(dico_ann)
        
        # Update info
        coco_dic['info']['date_created'] = str(date.today())
        coco_dic['info']['year'] = str(date.today().year)

        new_dic = {
            'info': coco_dic['info'],
            'licenses': coco_dic['licenses'],
            'images': images,
            'annotations': annotations,
            'categories': coco_dic['categories']
        }

        # Export json file
        with open(output_f, 'w') as outputfile:
            json.dump(new_dic, outputfile)

        if os.path.isfile(output_f) is True:
            print('File \'{}\' correctly saved at \'{}\'.'.format(file_name, sf_output_dir))
            print(' ')
        else:
            print('An error occurs, file \'{}\' not found at \'{}\'.'.format(file_name, sf_output_dir))

    return return_var
