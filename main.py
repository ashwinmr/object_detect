import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.models import load_model

def parse_args():
  """ Parse arguments for program
  """
  parser = argparse.ArgumentParser(description="Program for detection objects in an image")

  subparsers = parser.add_subparsers(required=True, dest='sub_command')

  # Create parser
  create_parser = subparsers.add_parser('create', help='Create dataset from image and groundtruth')
  create_parser.add_argument('image_path', help='path to image file')
  create_parser.add_argument('gt_path', help='path to ground truth xml')
  create_parser.add_argument('training_set_path', help='path to save training set')
  create_parser.add_argument('test_set_path', help='path to save test set')
  create_parser.set_defaults(func='create')

  # View parser
  view_parser = subparsers.add_parser('view', help='view data in a dataset')
  view_parser.add_argument('dataset_path', help='path to dataset')
  view_parser.set_defaults(func='view')

  # Propose parser
  prop_parser = subparsers.add_parser('propose', help='View region proposals for image')
  prop_parser.add_argument('image_path', help='path to image file')
  prop_parser.set_defaults(func='propose')

  # Train parser
  train_parser = subparsers.add_parser('train', help='train a model using a training set')
  train_parser.add_argument('dataset_path', help='path to dataset')
  train_parser.add_argument('model_path', help='path to save trained model')
  train_parser.set_defaults(func='train')

  # Eval parser
  eval_parser = subparsers.add_parser('eval', help='eval a model using a test set')
  eval_parser.add_argument('dataset_path', help='path to dataset')
  eval_parser.add_argument('model_path', help='path to saved model')
  eval_parser.set_defaults(func = 'eval')

  # Detect parser
  detect_parser = subparsers.add_parser('detect', help='detect objects in an image using a trained model')
  detect_parser.add_argument('image_path', help='path to image')
  detect_parser.add_argument('model_path', help='path to saved model')
  detect_parser.set_defaults(func = 'detect')

  return parser.parse_args()

def extract_boxes(annotation_file):
    """ Extract bounding boxes from an annotation file"""
    
    # Load and parse the file
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    
    # Get all bounding boxes
    boxes = []
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
        
    return boxes

def draw_bounding_boxes(ax,boxes,color = 'cyan'):
    """ Draw bounding boxes on axis
    """
    for box in boxes:
        width = box[2]-box[0]
        height = box[3] - box[1]
        xmin,ymin = box[0],box[1]
        rect = patches.Rectangle((xmin,ymin),width,height, edgecolor = color, fill=False)
        ax.add_patch(rect)

def show_probabilities(ax,coords,probabilities, y_offset = 0, color = 'red'):
    """ Show probabilities at coordinates on image
    """
    for i in range(len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        y -= y_offset
        ax.text(x,y,'P:{:.2f}'.format(probabilities[i]),color = color, fontsize = 'xx-small')
    
def get_iou(bb1, bb2):
    """ Determine Intersection over Union for two bounding boxes
    """

    # Determine the coordinates of the intersection rectangle
    xmin = max(bb1[0], bb2[0])
    ymin = max(bb1[1], bb2[1])
    xmax = min(bb1[2], bb2[2])
    ymax = min(bb1[3], bb2[3])

    if xmax < xmin or ymax < ymin:
        return 0.0

    # Determine intersection area
    intersection_area = (xmax - xmin) * (ymax - ymin)

    # Compute area of bounding boxes
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

def get_average_iou(boxes_detections, boxes_ground_truth):
    """ Determine the average IOU for a set of detections and ground truth boxes
    """
    total_iou = 0

    for box_dt in boxes_detections:
        max_iou = 0
        for box_gt in boxes_ground_truth:
            # Prevent multiple counting
            iou = get_iou(box_dt, box_gt)
            if iou > max_iou:
                max_iou = iou
        total_iou += max_iou

    average_iou = total_iou/len(boxes_detections)
    
    return average_iou

def gen_region_proposals(img, debug_plot=False):
    """ Generate region proposals for an image using selective search
    """

    # Selective search crashes on very large images
    # We will scale down the image and scale up the regions later
    height,width,_=img.shape
    scale = 1
    max_width = 1500
    if width > max_width:
        scale = width/max_width
        new_width = max_width
        new_height = int(height * new_width/ width)
        img_scaled = cv2.resize(img, (new_width, new_height))
    else:
        img_scaled = img

    # Perform selective search on scaled image
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_scaled)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    boxes = ss.process()

    # Scale back the resulting boxes and change results to xmin, ymin, xmax, ymax format
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        boxes[i] = x,y,x+w,y+h
        if scale != 1:
            boxes[i] = (boxes[i] * scale).astype(int)

    # Debug plot
    if debug_plot:
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        draw_bounding_boxes(ax,boxes,color = 'cyan')
        
        plt.show()

    return boxes

def create_dataset(image_path, gt_path, training_set_path, test_set_path, max_pos = 100, max_neg = 200, debug_plot = False):
    """ Create training and test set from image using selective search and bounding boxes of ground truth
    """

    # Load image
    main_image = cv2.imread(image_path)
    main_image = cv2.cvtColor(main_image,cv2.COLOR_BGR2RGB)

    # Load ground truth boxes
    boxes_gt = extract_boxes(gt_path)

    # Gen region proposals
    boxes_prop = gen_region_proposals(main_image, debug_plot=False)

    # Init positive and negative boxes
    boxes_pos = []
    boxes_neg = []

    # Find positive and negative region proposals using ground truth and iou
    for box_prop in boxes_prop:
        is_neg = True
        for box_gt in boxes_gt:
            iou = get_iou(box_prop, box_gt)
            # Prevent even slightly positive iou's being used as negative examples
            if iou > 0.1:
                is_neg = False
                if iou > 0.5:
                    boxes_pos.append(box_prop)
                break
        if is_neg:
            boxes_neg.append(box_prop)
    boxes_pos = np.array(boxes_pos)
    boxes_neg = np.array(boxes_neg)

    # Create dataset
    imgs = []
    labels = []

    for i,box in enumerate(boxes_pos):
        if i > max_pos:
            break
        xmin, ymin, xmax, ymax = box
        img = main_image[ymin:ymax,xmin:xmax]
        
        # Add scaled image as positive example
        img = cv2.resize(img,(100,100))
        imgs.append(img)
        labels.append(1)

    for i,box in enumerate(boxes_neg):
        if i > max_neg:
            break
        xmin, ymin, xmax, ymax = box
        img = main_image[ymin:ymax,xmin:xmax]
        
        # Add scaled image as negative  example
        img = cv2.resize(img,(100,100))
        imgs.append(img)
        labels.append(0)

    # Shuffle and split into training and test
    combined = list(zip(imgs,labels))
    random.shuffle(combined)
    imgs, labels = zip(*combined)
    split_idx = int(0.8*len(imgs))
    imgs_train = imgs[0:split_idx]
    labels_train = labels[0:split_idx]
    imgs_test = imgs[split_idx:]
    labels_test = labels[split_idx:]

    # Save the datasets

    data_train = {'images':imgs_train,'labels':labels_train}
    pickle.dump(data_train,open(training_set_path,"wb"))
    data_test = {'images':imgs_test,'labels':labels_test}
    pickle.dump(data_test,open(test_set_path,"wb"))

    # Debug plot
    if debug_plot:
        plt.imshow(main_image)
        plt.axis('off')
        ax = plt.gca()

        draw_bounding_boxes(ax,boxes_gt,color = 'cyan')
        draw_bounding_boxes(ax,boxes_pos,color = 'red')
        # draw_bounding_boxes(ax,boxes_neg,color = 'blue')

        plt.show()

def view_dataset(dataset_path):
  """ View data in a dataset
  """

  # Load dataset
  data = pickle.load(open(dataset_path,"rb"))

  imgs = data['images']
  labels = data['labels']

  # Create grid
  size = len(imgs)
  cols = math.ceil(np.sqrt(size))
  rows = math.ceil(size/cols)

  # Plot all images
  f, axarr = plt.subplots(rows,cols)

  for row in range(rows):
    for col in range(cols):
      idx = cols*row + col
      ax = axarr[row,col]
      ax.axis('off')
      if idx < size:
        ax.imshow(imgs[idx])
        ax.set_title(labels[idx],fontsize='x-small')

  plt.show()

  return

def normalize_images(imgs):
  """ Normalize images for learning
  """
  imgs_norm = imgs/255.0 - 0.5

  return imgs_norm

def denormalize_images(imgs_norm):
  """ De normalize images for plotting
  """
  imgs = (imgs_norm + 0.5) * 255.0

  return imgs

def get_dataset(dataset_path):
    """ Get dataset from a dataset file
    """
    # Load dataset
    data = pickle.load(open(dataset_path,"rb"))

    imgs = np.array(data['images'])
    labels = np.array(data['labels'])

    # Normalize images
    x = normalize_images(imgs)
    y = labels

    return x, y

def train_model(dataset_path, model_path):
    """ Train a model using a dataset
    """

    # Get training set
    x_train, y_train = get_dataset(dataset_path)

    # Create ML model
    model = Sequential([
        Flatten(input_shape=x_train[0].shape),
        Dense(1,activation='sigmoid'),
    ])

    # Train the model
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, validation_split = 0.2)

    # Save the model
    model.save(model_path)

def evaluate_model(dataset_path, model_path):
    """ evaluate the model in test mode using a dataset
    """
    # Load the trained model
    model = load_model(model_path)

    # Get data set
    x_test, y_test = get_dataset(dataset_path)

    # Evluate the model
    metrics = model.evaluate(x_test, y_test)

    # Display results
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))

    return

def detect_image(image_path, model_path):
    """ Use the pipeline and ML model to detect objects in an image
    """

    # Load the trained model
    model = load_model(model_path)

    # Load the image
    main_image = cv2.imread(image_path)
    main_image = cv2.cvtColor(main_image,cv2.COLOR_BGR2RGB)

    # Gen region proposals
    boxes_prop = gen_region_proposals(main_image, debug_plot=False)

    # Create image array for prediction
    imgs = []
    for box in boxes_prop:
        xmin, ymin, xmax, ymax = box
        img = main_image[ymin:ymax,xmin:xmax]
        img = cv2.resize(img,(100,100))
        img = normalize_images(img)
        imgs.append(img)
    imgs = np.array(imgs)

    # Predict
    pred = model.predict(imgs)

    # Compile detected boxes and probabilities
    boxes_det = []
    probabilities = []

    for i in range(len(imgs)):
        if pred[i] > 0.9:
            boxes_det.append(boxes_prop[i])
            probabilities.append(pred[i][0])

    # Plot
    plt.imshow(main_image)
    plt.axis('off')
    ax = plt.gca()

    # Draw ground truth bounding boxes
    boxes_gt = extract_boxes('GroundTruth.xml')
    draw_bounding_boxes(ax,boxes_gt,color = 'cyan')

    # Draw detection bounding boxes
    draw_bounding_boxes(ax,boxes_det,color = 'red')

    # Show probabilities
    coords = [[box[0], box[1]] for box in boxes_det]
    show_probabilities(ax,coords,probabilities, y_offset = 10)

    # Calculate average iou
    average_iou = get_average_iou(boxes_det, boxes_gt)
    ax.text(100,150,'Average IOU = {:.2f}'.format(average_iou),color = 'red')

    # Save to file
    plt.savefig('Output.png',dpi=300)

    plt.show()

def dummy_detection():
    """ Dummy detection pipeline
    """

    # Load image
    image_path = 'TestImage.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Plot image
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    # Draw ground truth bouding boxes
    boxes_gt = extract_boxes('GroundTruth.xml')
    draw_bounding_boxes(ax,boxes_gt,color = 'cyan')

    # Draw dummy detections bouding boxes
    boxes_dummy = extract_boxes('DummyDetections.xml')
    draw_bounding_boxes(ax,boxes_dummy,color = 'red')

    # Show dummy probabilities
    coords = [[box[0], box[1]] for box in boxes_dummy]
    probabilities = [0.5 for coord in coords]
    show_probabilities(ax,coords,probabilities, y_offset = 10)

    # Calculate average iou
    average_iou = get_average_iou(boxes_dummy, boxes_gt)
    ax.text(100,150,'Average IOU = {:.2f}'.format(average_iou),color = 'red')

    # Save to file
    plt.savefig('Output.png',dpi=300)

    plt.show()

if __name__ == "__main__":
    args = parse_args()

    if args.func == 'create':
        create_dataset(args.image_path, args.gt_path, args.training_set_path, args.test_set_path)
    if args.func == 'view':
        view_dataset(args.dataset_path)
    if args.func == 'propose':
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        gen_region_proposals(img,True)
    if args.func == 'train':
        train_model(args.dataset_path, args.model_path)
    if args.func == 'eval':
        evaluate_model(args.dataset_path, args.model_path)
    if args.func == 'detect':
        detect_image(args.image_path, args.model_path)
