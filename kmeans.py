from Fusion.utils.autoanchor import kmean_anchors


if __name__ == '__main__':
    """ 
    Change the path to the DATASET in the FLIR.yaml before running this script
    """
    anchor_num = 9
    _ = kmean_anchors(path='./FLIR.yaml', n=anchor_num, img_size=640, thr=4, gen=1000, verbose=True)