def get_cords(image_number):
    '''
    The coordinates are extracted from the Flir groundtruth labels.
    '''
    # x y w h
    imageNumber = "00007"
    bb_gtruth_00007 = []
    bb_gtruth_00007.append([225, 225, 17, 16])
    bb_gtruth_00007.append([281, 217, 42, 36])

    imageNumber = "00015"
    bb_gtruth_00015 = []
    bb_gtruth_00015.append([112, 219, 15, 13])
    bb_gtruth_00015.append([0, 237, 29, 21])
    bb_gtruth_00015.append([269, 235, 10, 19])

    imageNumber = "00026"
    bb_gtruth_00026 = []
    bb_gtruth_00026.append([328, 210, 111, 54])

    imageNumber = "00051"
    bb_gtruth_00051 = []
    bb_gtruth_00051.append([28, 244, 56, 39])
    bb_gtruth_00051.append([98, 245, 29, 23])
    bb_gtruth_00051.append([4, 240, 32, 28])

    imageNumber = "00070"
    bb_gtruth_00070 = []
    bb_gtruth_00070.append([281, 239, 21, 16])

    imageNumber = "01655"
    bb_gtruth_01655 = []
    bb_gtruth_01655.append([1, 228, 39, 59])

    imageNumber = '03670'
    bb_gtruth_03670 = []
    bb_gtruth_03670.append([46, 250, 170, 69])
    bb_gtruth_03670.append([7, 252, 74, 44])

    imageNumber = "04423"
    bb_gtruth_04423 = []
    bb_gtruth_04423.append([463, 223, 175, 115])
    bb_gtruth_04423.append([443, 257, 20, 47])
    bb_gtruth_04423.append([416, 214, 115, 79])
    bb_gtruth_04423.append([349, 218, 71, 52])
    bb_gtruth_04423.append([335, 227, 14, 32])
    bb_gtruth_04423.append([324, 235, 12, 16])
    bb_gtruth_04423.append([314, 226, 13, 13])
    bb_gtruth_04423.append([298, 233, 22, 21])
    bb_gtruth_04423.append([265, 229, 11, 10])
    bb_gtruth_04423.append([123, 213, 101, 78])

    imageNumber = "04431"
    bb_gtruth_04431 = [] 
    bb_gtruth_04431.append([475, 208, 20, 80])
    bb_gtruth_04431.append([497, 251, 17, 36])
    bb_gtruth_04431.append([108, 176, 367, 126])

    imageNumber = "04433"
    bb_gtruth_04433 = [] 
    bb_gtruth_04433.append([390, 205, 24, 79])
    bb_gtruth_04433.append([415, 250, 23, 29])
    bb_gtruth_04433.append([357, 252, 32, 29])
    bb_gtruth_04433.append([370, 216, 21, 31])
    bb_gtruth_04433.append([351, 213, 22, 37])
    bb_gtruth_04433.append([345, 221, 12, 27])
    bb_gtruth_04433.append([332, 221, 16, 22])
    bb_gtruth_04433.append([325, 222, 9, 13])
    bb_gtruth_04433.append([318, 221, 8, 13])
    bb_gtruth_04433.append([313, 224, 8, 10])
    bb_gtruth_04433.append([270, 221, 45, 34])
    bb_gtruth_04433.append([204, 215, 32, 40])
    bb_gtruth_04433.append([145, 217, 61, 47])
    bb_gtruth_04433.append([258, 209, 23, 27])

    imageNumber = "05856"
    bb_gtruth_05856 = [] 
    bb_gtruth_05856.append([248, 241, 18, 36])
    bb_gtruth_05856.append([232, 238, 20, 38])
    bb_gtruth_05856.append([556, 254, 28, 21])
    bb_gtruth_05856.append([459, 241, 114, 67])
    bb_gtruth_05856.append([357, 233, 127, 58])

    # corws=true!
    imageNumber = "07989"
    bb_gtruth_07989 = []
    bb_gtruth_07989.append([542, 218, 97, 77])
    bb_gtruth_07989.append([596, 230, 43, 80])
    bb_gtruth_07989.append([406, 217, 9, 22])

    imageNumber = "08566"
    bb_gtruth_08566 = []
    bb_gtruth_08566.append([371, 245, 23, 19])

    imageNumber = "08637"
    bb_gtruth_08637 = []
    bb_gtruth_08637.append([78, 234, 58, 42])
    bb_gtruth_08637.append([284, 213, 15, 12])
    bb_gtruth_08637.append([559, 224, 27, 33])
    bb_gtruth_08637.append([341, 220, 21, 16])

    imageNumber = "08862"
    bb_gtruth_08862 = [] 
    bb_gtruth_08862.append([5, 246, 143, 90])
    bb_gtruth_08862.append([288, 221, 18, 16])
    bb_gtruth_08862.append([335, 225, 34, 30])
    bb_gtruth_08862.append([326, 226, 17, 21])
    bb_gtruth_08862.append([29, 234, 14, 40])
    bb_gtruth_08862.append([38, 234, 21, 30])
    bb_gtruth_08862.append([621, 242, 18, 21])

    ####################### Train set finishes at 8862 and Val set starts at 8863 which is annotated as 0 in jason file

    # Val_Set
    imageNumber = "09833"
    bb_gtruth_09833 = [] 
    bb_gtruth_09833.append([36, 239, 39, 122]) # FLIR_09833 | 09833-8863 = 970 in anotation file!
    bb_gtruth_09833.append([65, 240, 24, 113])
    bb_gtruth_09833.append([609, 219, 29, 110])
    bb_gtruth_09833.append([159, 275, 132, 31])

    dict_gth = {
        # 1800 * 1600
        '00007': bb_gtruth_00007,
        '00015': bb_gtruth_00015,
        '00026': bb_gtruth_00026,
        '00051': bb_gtruth_00051,
        '00070': bb_gtruth_00070,
        '01655': bb_gtruth_01655,
        '03670': bb_gtruth_03670, # night
        '04423': bb_gtruth_04423,
        '04431': bb_gtruth_04431,
        '04433': bb_gtruth_04433,
        # ValSet
        '09833': bb_gtruth_09833,

        # 2048 * 1536
        '05856': bb_gtruth_05856,
        '07989': bb_gtruth_07989,
        '08566': bb_gtruth_08566,

        # 1280 * 1024
        '08637': bb_gtruth_08637,
        '08862': bb_gtruth_08862,
    }

    bb_gtruth = dict_gth.get(image_number)

    return bb_gtruth