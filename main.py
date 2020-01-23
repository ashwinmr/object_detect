import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

if __name__ == "__main__":

    # Load imge
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
    ax.text(100,200,'Average IOU = {:.2f}'.format(average_iou),color = 'red')

    # Save to file
    plt.savefig('Output.png',dpi=300)

    plt.show()
