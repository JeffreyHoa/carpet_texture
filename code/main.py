# Copyright 2017 Jeffrey Hoa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import time

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../image/1.png'
    print(__doc__)

    img = cv2.imread(fn, True)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed_pt = None
    fixed_range = True
    connectivity = 4

##########################################################

    BACK_SLASH    = 0
    FORWARD_SLASH = 1
    DIRECT_POS = 0
    DIRECT_NEG = 1

    # It should be within ths range of photo.
    # (1) backslash    (2) forwardslash
    # ------------> x  ------------> x
    # |    /neg        |  \neg
    # |   /            |   \
    # |  /pos          |    \pos
    # y                y
    #
    # rulerRange:
    # nearly half of len of ruler excluding the center pt.
    #
    def ruler_measureResult(grayImg, startVec, ruler, neg_stepsLimit=200, pos_stepsLimit=200):
        height, width = grayImg.shape[:2]

        # Return variables
        neg_vecPt = np.zeros(2)
        pos_vecPt = np.zeros(2)

        # Set incVec: from center to ends.
        if ruler == BACK_SLASH:
            negIncVec = np.array([ 1,-1])
            posIncVec = np.array([-1, 1])
        else:
            negIncVec = np.array([-1,-1])
            posIncVec = np.array([ 1, 1])


        # Jeffrey: should be within size of img.
        # steps_limit = 200

        # Measure steps in neg direction.
        now_vecPt = startVec
        for idx in range(1,neg_stepsLimit+1):
            now_vecPt = now_vecPt+negIncVec
            now_x, now_y = now_vecPt

            if grayImg[now_y][now_x] == 255:
                # cv debug
                grayImg[now_y][now_x] = 255
            else:
                # Go back one step.
                neg_vecPt = now_vecPt-negIncVec
                print("idx = ", idx);
                break;

            if idx == neg_stepsLimit:
                neg_vecPt = now_vecPt
                break;

        # Measure steps in pos direction.
        now_vecPt = startVec
        for idx in range(1,pos_stepsLimit+1):
            now_vecPt = now_vecPt+posIncVec
            now_x, now_y = now_vecPt

            if grayImg[now_y][now_x] == 255:
                # cv debug
                grayImg[now_y][now_x] = 255
            else:
                # Go back one step.
                pos_vecPt = now_vecPt-posIncVec
                print("idx = ", idx);
                break;

            if idx == pos_stepsLimit:
                pos_vecPt = now_vecPt
                break;

        return (neg_vecPt, pos_vecPt)



    def ruler_nextEndLimit(negEnd_vecPt, posEnd_vecPt, ruler, direct):
        #           |------|
        # neg_pt |------------| pos_pt

        if ruler == FORWARD_SLASH:
            negIncVec = np.array([ 1,-1])
            posIncVec = np.array([-1, 1])
        else:
            negIncVec = np.array([-1,-1])
            posIncVec = np.array([ 1, 1])

        # Result variables.
        next_negEnd_vecPt_limit = np.zeros(2)
        next_posEnd_vecPt_limit = np.zeros(2)


        # This is a basic 45-degree of limit.
        # we may find a better next_neg_pt & next_pos_pt.
        #
        # Jeffrey: should be within size of img.
        if direct == DIRECT_NEG:
            next_negEnd_vecPt_limit = negEnd_vecPt + negIncVec
            next_posEnd_vecPt_limit = posEnd_vecPt + negIncVec
        else:
            next_negEnd_vecPt_limit = negEnd_vecPt + posIncVec
            next_posEnd_vecPt_limit = posEnd_vecPt + posIncVec


        return next_negEnd_vecPt_limit, next_posEnd_vecPt_limit



    def ruler_nextStartPoint(next_neg_vecPt_limit, next_pos_vecPt_limit):

        incVec_toNextCenterX = (next_pos_vecPt_limit[0] - next_neg_vecPt_limit[0]) // 2
        incVec_toNextCenterY = (next_pos_vecPt_limit[1] - next_neg_vecPt_limit[1]) // 2

        if abs(incVec_toNextCenterX) is not abs(incVec_toNextCenterY):
            if incVec_toNextCenterY > 0:
                incVec_toNextCenterY = abs(incVec_toNextCenterX)
            else:
                incVec_toNextCenterY = 0-abs(incVec_toNextCenterX)

        incVec_toNextCenter = np.array([incVec_toNextCenterX, incVec_toNextCenterY])

        next_start_vecPt = next_neg_vecPt_limit + incVec_toNextCenter

        print("next_start_vecPt: ", next_start_vecPt)
        return next_start_vecPt



    def update(dummy=None):
        if seed_pt is None:
            cv2.imshow('floodfill', img)
            return

        flooded = img.copy()
        mask[:] = 0
        lo = cv2.getTrackbarPos('lo', 'floodfill')
        hi = cv2.getTrackbarPos('hi', 'floodfill')
        flags = connectivity
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE

        ########################
        # 1. Cluster: FloodFill
        ########################
        cv2.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)


        ########################
        # 2. Graylize
        ########################
        # 2.1 transform to gray_flooded
        gray_flooded = cv2.cvtColor(flooded, cv2.COLOR_BGR2GRAY)

        # 2.2 get its coresponding param.
        gray_x, gray_y = seed_pt
        gray_seed_pt = gray_x, gray_y

        # 2.3 Debug:
        print("Click pt:", gray_seed_pt, "Ori Color: ", gray_flooded[gray_y][gray_x])
        gray_flooded[gray_y][gray_x] = 255


        ########################
        # 3. Init ruler.
        ########################
        # Get positions of both the negPt end and posPt.
        # We set: direction to x coordinate is negative.
        # Initialy, we have two rulers as following.
        # (1) backslash    (2) forwardslash
        # ------------> x  ------------> x
        # |    /neg        |  \neg
        # |   /            |   \
        # |  /pos          |    \pos
        # y                y

        ruler_list = [BACK_SLASH, FORWARD_SLASH]
        direct_list = [DIRECT_NEG, DIRECT_POS]
        for ruler in ruler_list:
            for direct in direct_list:

                # Vectorization.
                x, y = gray_seed_pt
                gray_seed_vecPt = np.array([x, y])
                gray_seed_vecPt.astype(int)

                neg_vecPt, pos_vecPt = ruler_measureResult(gray_flooded, gray_seed_vecPt, ruler)
                print("[Init] neg_pt:", neg_vecPt, "pos_pt:", pos_vecPt)


                ###################################
                # 4. Traverse and filter out noise.
                ###################################
                steps = 200

                for idx in range(1, steps+1):

                    # 4.1 next pos and neg limit ends.
                    next_neg_vecPt_limit, next_pos_vecPt_limit = \
                            ruler_nextEndLimit(neg_vecPt, pos_vecPt, ruler, direct)

                    # 4.2 next start center point.
                    next_start_vecPt = \
                            ruler_nextStartPoint(next_neg_vecPt_limit, next_pos_vecPt_limit)

                    # 4.3 Update pos and neg avaliable ends.
                    neg_stepsLimit = abs(next_neg_vecPt_limit[0] - next_start_vecPt[0])
                    pos_stepsLimit = abs(next_pos_vecPt_limit[0] - next_start_vecPt[0])

                    neg_vecPt, pos_vecPt = \
                            ruler_measureResult(gray_flooded, next_start_vecPt, ruler, \
                            neg_stepsLimit, pos_stepsLimit)

                    print("[Next] neg_pt:", neg_vecPt, "pos_pt:", pos_vecPt)





                    # Break the loop.
                    if abs(neg_vecPt[0]-pos_vecPt[0]) == 1:
                        cv2.circle(flooded, (int(pos_vecPt[0]), int(neg_vecPt[1])), 2, (255,0,0), -1)
                        print("Corner:",  (pos_vecPt[0], neg_vecPt[1]))
                        cv2.imshow('gray_floodfill', gray_flooded)
                        cv2.imshow('floodfill', flooded)

                        break;
                    if abs(neg_vecPt[0]-pos_vecPt[0]) == 0:
                        cv2.circle(flooded, (int(pos_vecPt[0]), int(neg_vecPt[1])), 2, (0,255,0), -1)
                        print("Corner:",  (pos_vecPt[0], neg_vecPt[1]))
                        cv2.imshow('gray_floodfill', gray_flooded)
                        cv2.imshow('floodfill', flooded)
                        break;
                    else:
                        cv2.imshow('floodfill', gray_flooded)
                        continue;



    def onmouse(event, x, y, flags, param):
        global seed_pt
        if flags & cv2.EVENT_FLAG_LBUTTON:
            start = time.time()

            seed_pt = x, y
            print("=================== Start ===================")
            print("size: w,h = ", w,h)
            print("seed_pt = ", seed_pt)
            update()

            elapsed = (time.time() - start)
            print("Time used:", elapsed)



    update()
    cv2.setMouseCallback('floodfill', onmouse)
    cv2.createTrackbar('lo', 'floodfill', 200, 255, update)
    cv2.createTrackbar('hi', 'floodfill', 40, 255, update)

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord('f'):
            fixed_range = not fixed_range
            print('using %s range' % ('floating', 'fixed')[fixed_range])
            update()
        if ch == ord('c'):
            connectivity = 12-connectivity
            print('connectivity =', connectivity)
            update()
    cv2.destroyAllWindows()
