cimport cython
import numpy as np
cimport numpy as np
import cv2
from libc.stdlib cimport calloc,free
from cpython.mem cimport PyMem_Malloc as malloc
from cpython.mem cimport PyMem_Free as freep

from cpython.array cimport array
ctypedef np.float_t FTYPE_t
ctypedef np.uint8_t CTYPE_t
ctypedef np.int64_t LTYPE_t
ctypedef np.uint32_t DTYPE_t

# creates pointer to memory stack and initiates all values to 0
cdef bint *tablelut_ptr = <bint *>calloc(256*256*256,sizeof(bint))
mtx = np.array([[ 620.54646,0, 316.17234],[0,621.10631,244.57960],[0,0,1]],dtype=np.float64)
dst = np.array([ 0.13359,-0.29557, -0.00070 ,-0.00054,0.00000 ],dtype=np.float64)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void tablegen(long hmin,long hmax,double m1, double c1, double m2, double c2, double thresh):
    cdef:
        #unsigned char[:,:,::1] table = np.zeros((256,256,256),dtype = np.uint8)
        #unsigned char* table_ptr = &table[0,0,0]
        long x0,y0,z0
        double b,g,r,saturation
        int K
        double chroma
        long hue
    global tablelut_ptr        
    for x0 in range(256):
        for y0 in range(256):
            for z0 in range(256):
                b,g,r = x0,y0,z0
                K = 0
                if g < b:
                    g,b = b,g
                    K = -6
                if r < g:
                    r,g = g,r
                    K =  -K - 2
                chroma = r - min(g,b)
                if r != 0:
                    saturation = 255 * chroma/(r * 1.0)
                else:
                    saturation = 0.0
                if chroma != 0:
                    hue = int(30 * abs(K +((g-b)/(chroma))))
                    if (hue>= hmin) & (hue <= hmax) & (r >= thresh):
                        if (r >=  (m1 * saturation) + c1) & (r <=  (m2 * saturation) + c2):
                                tablelut_ptr[(256*256*x0) + (256*y0)+z0] = 1
    #return None

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef unsigned char[:,:,::1] tablegen2(long hmin,long hmax,long smin, long smax, long v, long vmax):
    cdef:
        unsigned char[:,:,::1] table = np.zeros((256,256,256),dtype = np.uint8)
        long x0,y0,z0
        double b,g,r
        int K
        double chroma
        long hue
        long saturation
        unsigned char* table_ptr = &table[0,0,0]        
    for x0 in range(256):
        for y0 in range(256):
            for z0 in range(256):
                b,g,r = x0,y0,z0
                K = 0
                if g < b:
                    g,b = b,g
                    K = -6
                if r < g:
                    r,g = g,r
                    K =  -K - 2
                chroma = r - min(g,b)
                if r != 0:
                    saturation = int(255 * chroma/(r * 1.0))
                else:
                    saturation = 0
                if chroma != 0:
                    hue = int(30 * abs(K +((g-b)/(chroma))))
                    if (hue>= hmin) & (hue <= hmax) & (r >=v) & (r <= vmax) & (saturation >= smin) & (saturation <= smax):
                        table_ptr[(256*256*x0) + (256*y0)+z0] = 1
    return table
 

srcarray =  1.0 * np.mgrid[0:640,0:480].T.reshape(1,480*640,2)
cdef double[:,:,::1] undistortptr = cv2.undistortPoints(srcarray,mtx,dst,R = None,P= mtx)
#cdef double* undistortptr= &result[0,0]

#cdef unsigned char[:,:,::1] tablelut = tablegen2(87,90,50,255,190,255)
#cdef unsigned char[:,:,::1] tablelut = tablegen2(163,179,50,255,50,255)
##cdef unsigned char[:,:,::1] tablelut = tablegen2(40,80,5,255,60,255) #old threshold system

#####cdef unsigned char[:,:,::1] tablelut = tablegen(40,80,-1.0,100.0,-0.651162,242.7906,10.0) #consider changing thresh to 15.0 or 20.0 
#####cdef unsigned char* tablelut_ptr = &tablelut[0,0,0]

tablegen(40,80,-1.0,100.0,-0.651162,242.7906,40.0) #consider changing thresh to >40

#cdef long[::1] x_out = np.zeros((480*640), dtype = np.int32)
#cdef long* x_outptr = &x_out[0]
#cdef long[::1] y_out = np.zeros((640), dtype = np.int32)
#cdef long* y_outptr = &y_out[0]

#rect contains coordinates of square. Qinv is inverse of np.dot(mtx,R)
def final_return(rect,Qinv):
    ##cdef:
    ##   double[::1] outval = np.array([0,0,0],dtype = np.float64) #initialise ouput array

    xcord = int((rect[2] + rect[3])/2) #find x pixel
    ycord = int((rect[0] + rect[1])/2) #find y pixel

    #undistort pixels
    res = undistortptr[0,(xcord*640) + ycord]
    #print "res is",np.asarray(res)
    ppos = np.array([[res[0]],[res[1]],[1]])
    #print "ppos is",np.asarray(ppos) 

    #calculate u vector 3 x 1
    u = np.dot(Qinv,ppos)

    return u


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def squarelut6(long[::1] input,long x, long y,unsigned char v,unsigned char[:,:,::1] image): ###, bint *tablelut_ptr1=<bint *>tablelut_ptr)
    cdef:
        long* inputptr= &input[0]
        long xminscan = (inputptr[2]<= inputptr[3])*((v+1)*inputptr[2] > (v*inputptr[3]))*((v + 1)*inputptr[2] - (v*inputptr[3]))
        long yminscan = (inputptr[0]<= inputptr[1])*((v+1)*inputptr[0] > (v*inputptr[1]))*((v + 1)*inputptr[0] - (v*inputptr[1]))
        long xmaxscan = ((inputptr[2]<= inputptr[3])*((v+1)*(inputptr[3]+1) < x + (v*inputptr[2]))*(((v+1)*(inputptr[3]+1)) - (v*inputptr[2]) - x)) + x
        long ymaxscan = ((inputptr[0]<= inputptr[1])*((v+1)*(inputptr[1]+1) < y + (v*inputptr[0]))*(((v+1)*(inputptr[1]+1)) - (v*inputptr[0]) - y)) + y
        long deltax = xmaxscan - xminscan
        long deltay = ymaxscan - yminscan
        unsigned char* img_ptr = &image[0,0,0]
        unsigned char inc
        bint *tablelut_ptr1=tablelut_ptr
        
        #long[::1] x_out = np.zeros((deltax*deltay), dtype = np.int32)
        #long* x_outptr = &x_out[0]
        long *x_outptr =<long *>malloc(deltax*deltay*sizeof(long))
        #long[::1] y_out = np.zeros((deltay), dtype = np.int32)
        #long* y_outptr = &y_out[0]
        long *y_outptr =<long *>malloc(deltay*sizeof(long))
        
        long[::1] outputptr = np.array([y,0,0,0], dtype = np.int32)
        #long* outputptr = &output[0]
        #long yrmin = y
        #long yrmax = 0
        #long xrmin, xrmax
        
        long x0,y0,i0
        long i = 0
        
        #long pos[3]
        #long* posptr = &pos[0]
        long *posptr =<long *>malloc(3*sizeof(long))
    posptr[0] = (xminscan*y) +yminscan
    y_outptr[1] = y
    for x0 in range(deltax):
        posptr[1] = posptr[0] + (x0 *y)
        i0 = i
        for y0 in range(deltay):
            posptr[2] = 3 * (posptr[1] + y0)
            inc = tablelut_ptr1[256*256*img_ptr[posptr[2]] + 256*img_ptr[posptr[2] +1] + img_ptr[posptr[2] + 2]]
            #inc = tablelut_ptr[256*256*img_ptr[3*((xminscan + x0)*y + y0 + yminscan)] + 256*img_ptr[3*((xminscan + x0)*y + y0 + yminscan) +1] + img_ptr[3*((xminscan + x0)*y + y0 + yminscan) + 2]]
            i += inc
            y_outptr[inc*(i - i0)] = y0 + yminscan
            x_outptr[inc*i] = x0 + xminscan
        #break out of loop if no pixel of ineterest detected in 3 lines
        if x0 - x_outptr[i] + xminscan > 2:
            break
        y_outptr[0] = 0
        outputptr[0] += (y_outptr[1] - outputptr[0])*(outputptr[0] > y_outptr[1])
        outputptr[1] += (y_outptr[i-i0] - outputptr[1])*(outputptr[1] < y_outptr[i - i0])
        #yrmin += (y_outptr[1] - yrmin)*(yrmin > y_outptr[1])
        #yrmax += (y_outptr[i-i0] - yrmax)*(yrmax < y_outptr[i - i0])
    outputptr[2] = x_outptr[1]
    outputptr[3] = x_outptr[i]
    freep(x_outptr)
    freep(posptr)
    freep(y_outptr)
    return outputptr


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def squarelut8(int[::1] output,int x, int y,unsigned char v,unsigned char[:,:,::1] image): ###, bint *tablelut_ptr1=<bint *>tablelut_ptr)
    cdef:
        int* outputptr = &output[0]
        unsigned char* img_ptr = &image[0,0,0]
        
        int yminscan = (outputptr[0]<= outputptr[1])*((v+1)*outputptr[0] > (v*outputptr[1]))*((v + 1)*outputptr[0] - (v*outputptr[1]))
        int ymaxscan = ((outputptr[0]<= outputptr[1])*((v+1)*(outputptr[1]+1) < y + (v*outputptr[0]))*(((v+1)*(outputptr[1]+1)) - (v*outputptr[0]) - y)) + y
        int deltay = ymaxscan-yminscan
        int deltax = outputptr[5] - outputptr[4]
        bint inc
        bint xpos = 0
        bint *tablelut_ptr1=tablelut_ptr

        int *x_outptr =<int *>malloc(2*sizeof(int))

        int *y_outptr =<int *>malloc(deltay*sizeof(int))
        
        #int[::1] output = array('i',[ymaxscan-1,0,0,0,0,0])
        #int* outputptr = &output[0]
        
        int x0,y0,i0
        int i = 0
        
        long *posptr =<long *>malloc(2*sizeof(long))
    outputptr[0] = ymaxscan-1
    outputptr[1] = 0
    
    x_outptr[0] = deltax
    x_outptr[1] = deltax - 1
    
    y_outptr[1] = ymaxscan - 1 
    for x0 in range(deltax):
        posptr[0] = (x0 *y)
        i0 = i
        for y0 in range(yminscan,ymaxscan,3 - (2*xpos)):
            posptr[1] = 3 * (posptr[0] + y0)
            inc = tablelut_ptr1[256*256*img_ptr[posptr[1]] + 256*img_ptr[posptr[1] +1] + img_ptr[posptr[1] + 2]]
            i += inc
            y_outptr[inc*(i - i0)] = y0
        #i > i0 if row contains a valid pixel. add 0 contains first value. add 1 contains latest value
        if i > i0:
            x_outptr[xpos] = x0
            xpos = 1
            outputptr[0] += (y_outptr[1] - outputptr[0])*(outputptr[0] > y_outptr[1])
            outputptr[1] += (y_outptr[i-i0] - outputptr[1])*(outputptr[1] < y_outptr[i - i0])
            # change search area 
            ymaxscan += (outputptr[1] + 5 -ymaxscan) * (outputptr[1] + 5 < ymaxscan) 
        #break out of loop if no pixel of ineterest detected in 3 lines after xpos switched on
        elif x0 - x_outptr[1] > 2:
            break        
    outputptr[2] = x_outptr[0] + outputptr[4] #if none detected outputptr[2] = deltax + image start
    outputptr[3] = x_outptr[1] + outputptr[4] #if none detected outputptr[3] = delatx - 1 + imagestart
    #if none detected outputptr[4] = 0 and outputptr[5] = x
    #if search area crosses x or becomes negative second condition switches off    
    outputptr[4] = (outputptr[2]<= outputptr[3])  *  (((v+1)*outputptr[2]) > (v*outputptr[3]))  *  ((v + 1)*outputptr[2] - (v*outputptr[3]))
    outputptr[5] = ((outputptr[2]<= outputptr[3]) *  ((v+1)*(outputptr[3]+1) < x + (v*outputptr[2]))  * (((v+1)*(outputptr[3]+1)) - (v*outputptr[2]) - x)) + x
    freep(x_outptr)
    freep(y_outptr)
    freep(posptr)
    return output

#used by binary tester
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsv(unsigned char[:,:,::1] image,long x, long y):
    cdef:
        unsigned char[:,:,::1] output = np.zeros((x,y,3),dtype = np.uint8)
        long x0,y0
        double b,g,r
        double chroma
    for x0 in range(x):
        for y0 in range(y):
            b = image[x0,y0,0]
            g = image[x0,y0,1]
            r = image[x0,y0,2]
            K = 0
            if g < b:
                g,b = b,g
                K = -6
            if r < g:
                r,g = g,r
                K =  -K - 2
            chroma = r - min(g,b)
            if chroma != 0:
                output[x0,y0,0] = int(30 * abs(K +((g-b)/(chroma))))
            if r != 0:
                output[x0,y0,1] = int(255* chroma/(r * 1.0))
            output[x0,y0,2] = int(r)
    return output

#used by calibhue and calibmouse1   
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsvarray2(unsigned char[:,:,::1] image,long xmin,long xmax,long ymin, long ymax):
    cdef:
        unsigned char* img_ptr = &image[0,0,0]
        unsigned long[:,::1] yaxis = np.zeros((3,256),dtype = np.uint32)
        unsigned long* yaxisptr = &yaxis[0,0]
        long x0,y0
        double b,g,r
        long hue,saturation
        double chroma
        long deltax = xmax - xmin
        long deltay = ymax - ymin
    for x0 in range(deltax):
        for y0 in range(deltay):
            b = img_ptr[3*((xmin +x0)* 640 + y0 + ymin)]
            g = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 1]
            r = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 2]
            K = 0
            if g < b:
                g,b = b,g
                K = -6
            if r < g:
                r,g = g,r
                K =  -K - 2
            chroma = r - min(g,b)
            if chroma != 0:
                hue = int(30 * abs(K +((g-b)/(chroma))))
            if r != 0:
                saturation = int(255* chroma/(r * 1.0))
            else:
                saturation = 0
            yaxisptr[hue] += 1
            yaxisptr[256 + saturation] += 1
            yaxisptr[256*2 + int(r)] += 1
    return yaxis

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsvarray3(unsigned char[:,:,::1] img_ptr,long x=480,long y=640):
    cdef:
        #unsigned char* img_ptr = &image[0,0,0]
        unsigned long[:,:,::1] colptr = np.zeros((256,256,256),dtype = np.uint32)
        #unsigned long* colptr = &colourscat[0,0,0]
        long x0,y0
        double b,g,r
        long hue,saturation,val
        double chroma
    for x0 in range(x):
        for y0 in range(y):
            b = img_ptr[x0,y0,0]
            g = img_ptr[x0,y0,1]
            r = img_ptr[x0,y0,2]
            K = 0
            if g < b:
                g,b = b,g
                K = -6
            if r < g:
                r,g = g,r
                K =  -K - 2
            chroma = r - min(g,b)
            if chroma != 0:
                hue = int(30 * abs(K +((g-b)/(chroma))))
            if r != 0:
                saturation = int(255* chroma/(r * 1.0))
            else:
                saturation = 0
            val = int(r)
            colptr[hue,saturation,val] += 1
    return colptr

#used by calibhuemouse3d
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsvarray3c(unsigned char[:,:,::1] image,long xmin,long xmax,long ymin, long ymax):
    cdef:
        unsigned char* img_ptr = &image[0,0,0]
        unsigned long[:,:,::1] colourscat = np.zeros((256,256,256),dtype = np.uint32)
        unsigned long* colptr = &colourscat[0,0,0]
        long x0,y0
        double b,g,r
        long hue,saturation
        double chroma
        long deltax = xmax - xmin
        long deltay = ymax - ymin
    for x0 in range(deltax):
        for y0 in range(deltay):
            b = img_ptr[3*((xmin +x0)* 640 + y0 + ymin)]
            g = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 1]
            r = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 2]
            K = 0
            if g < b:
                g,b = b,g
                K = -6
            if r < g:
                r,g = g,r
                K =  -K - 2
            chroma = r - min(g,b)
            if chroma != 0:
                hue = int(30 * abs(K +((g-b)/(chroma))))
            if r != 0:
                saturation = int(255* chroma/(r * 1.0))
            else:
                saturation = 0
            colptr[(256*256*hue) + (256*saturation)+int(r)] += 1
    return colourscat
  
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def lineplotter(np.ndarray saturation, np.ndarray val, int valthresh):
    assert saturation.dtype == np.int and val.dtype == np.int
    cdef:
      np.ndarray newsat = saturation[np.where(val[:] >= valthresh)]
      np.ndarray newval = val[np.where(val[:] >= valthresh)]
      np.ndarray carray = newsat + newval
      np.ndarray outputmin = 2.0 * np.ones(len(newsat),dtype = np.float64)
      np.ndarray outputmax = 2.0 * np.ones(len(newsat),dtype = np.float64)
      int posmin = np.where(carray == min(carray))[0][0]
      int posmax = np.where(carray == max(carray))[0][0]
      int i, maxlinepos,minlinepos
      np.ndarray outarray = np.zeros(4, dtype = np.float64)
    for i in range(len(carray)):
      if newsat[i] != newsat[posmin]:
        outputmin[i] = abs(((newval[i] - newval[posmin])/(1.0*(newsat[i] - newsat[posmin]))) + 1.0 )
      if newsat[i] != newsat[posmax]:
        outputmax[i] = abs(((newval[i] - newval[posmax])/(1.0*(newsat[i] - newsat[posmax]))) + 1.0 )
    maxlinepos = np.where(outputmax == min(outputmax))[0][0]
    outarray[2] = (newval[maxlinepos] - newval[posmax])/(1.0 * (newsat[maxlinepos] - newsat[posmax]))
    outarray[3] = (newsat[maxlinepos]*newval[posmax] - newval[maxlinepos]*newsat[posmax])/(1.0 * (newsat[maxlinepos] - newsat[posmax]))
    minlinepos = np.where(outputmin == min(outputmin))[0][0]
    outarray[0] = (newval[minlinepos] - newval[posmin])/(1.0 * (newsat[minlinepos] - newsat[posmin]))
    outarray[1] = (newsat[minlinepos]*newval[posmin] - newval[minlinepos]*newsat[posmin])/(1.0 * (newsat[minlinepos] - newsat[posmin]))
    return outarray

#used by calibhuemouselaser
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsvarrayl(unsigned char[:,:,::1] backi, unsigned char[:,:,::1] inputi,long xmin,long xmax,long ymin, long ymax,double thresh):
    cdef:
        #unsigned char[:,:,::1] hsvbacki = cv2.cvtColor(np.asarray(backi[xmin:xmax,ymin:ymax]), cv2.COLOR_BGR2HSV)
        #unsigned char[:,:,::1] hsvinputi = cv2.cvtColor(np.asarray(inputi[xmin:xmax,ymin:ymax]), cv2.COLOR_BGR2HSV)
        unsigned char[:,:,::1] diffimage = cv2.absdiff(np.asarray(backi[xmin:xmax,ymin:ymax]),np.asarray(inputi[xmin:xmax,ymin:ymax]))
        unsigned char* img_ptr = &diffimage[0,0,0]
        unsigned long[:,:,::1] colourscat = np.zeros((256,256,256),dtype = np.uint32)
        unsigned long* colptr = &colourscat[0,0,0]
        long x0,y0
        double b,g,r
        long hue,saturation
        double chroma
        long deltax = xmax - xmin
        long deltay = ymax - ymin
        unsigned char pix0,pix1,pix2
        
        unsigned char[:,::1] outarray = np.zeros((deltax,deltay), dtype = np.uint8)
        unsigned char* outptr = &outarray[0,0]
        
        double normval
        #double threshold = 100.00
        
    for x0 in range(deltax):
        for y0 in range(deltay):
            pix0 = img_ptr[3*(x0*deltay + y0)]
            pix1 = img_ptr[3*(x0*deltay + y0)+1]
            pix2 = img_ptr[3*(x0*deltay + y0)+2]
            normval = ((pix0**2.0)+(pix1**2.0)+(pix2**2.0))**0.5
            if normval > thresh:
                outptr[x0*deltay + y0] = 255
            else:
                outptr[x0*deltay + y0] = 0
            
            #b = img_ptr[3*((xmin +x0)* 640 + y0 + ymin)]
            #g = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 1]
            #r = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 2]
            #K = 0
            #if g < b:
            #    g,b = b,g
            #    K = -6
            #if r < g:
            #    r,g = g,r
            #    K =  -K - 2
            #chroma = r - min(g,b)
            #if chroma != 0:
            #    hue = int(30 * abs(K +((g-b)/(chroma))))
            #if r != 0:
            #    saturation = int(255* chroma/(r * 1.0))
            #else:
            #    saturation = 0
            #colptr[(256*256*hue) + (256*saturation)+int(r)] += 1
    return outarray

#used by calibhuemouselaserex
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def bgrhsvarraylc(unsigned char[:,::1] diffimage, unsigned char[:,:,::1] inputi,long xmin,long xmax,long ymin, long ymax,double thresh):
    cdef:
        unsigned char* diff_ptr = &diffimage[0,0]
        unsigned char* img_ptr = &inputi[0,0,0]
        
        unsigned long[:,:,::1] colourscat = np.zeros((256,256,256),dtype = np.uint32)
        unsigned long* colptr = &colourscat[0,0,0]
        
        long x0,y0
        double b,g,r
        long hue,saturation
        double chroma
        
        long deltax = xmax - xmin
        long deltay = ymax - ymin
        
        #unsigned char pix0,pix1,pix2
        #double threshold = 100.00
        
    for x0 in range(deltax):
        for y0 in range(deltay):
            if diff_ptr[x0*deltay + y0] == 255:
                b = img_ptr[3*((xmin +x0)* 640 + y0 + ymin)]
                g = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 1]
                r = img_ptr[3*((xmin +x0)* 640 + y0 + ymin) + 2]
                K = 0
                if g < b:
                        g,b = b,g
                        K = -6
                if r < g:
                        r,g = g,r
                        K =  -K - 2
                chroma = r - min(g,b)
                if chroma != 0:
                        hue = int(30 * abs(K +((g-b)/(chroma))))
                if r != 0:
                        saturation = int(255* chroma/(r * 1.0))
                else:
                        saturation = 0
                colptr[(256*256*hue) + (256*saturation)+int(r)] += 1
    return colourscat

def cleanupf():
        global tablelut_ptr
        free(tablelut_ptr)
