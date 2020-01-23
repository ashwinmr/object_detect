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
        ax.text(x,y,'P:{:.2f}'.format(probabilities[i]),color = color, fontsize = 'x-small')
    
if __name__ == "__main__":

    # Load imge
    image_path = 'TestImage.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Plot image
    plt.imshow(img)
    ax = plt.gca()

    # Draw ground truth bouding boxes
    boxes = extract_boxes('GroundTruth.xml')
    draw_bounding_boxes(ax,boxes,color = 'cyan')

    # Draw dummy detections bouding boxes
    boxes = extract_boxes('DummyDetections.xml')
    draw_bounding_boxes(ax,boxes,color = 'red')

    # Show dummy probabilities
    coords = [[box[0], box[1]] for box in boxes]
    probabilities = [0.5 for coord in coords]
    show_probabilities(ax,coords,probabilities, y_offset = 10)

    plt.show()
