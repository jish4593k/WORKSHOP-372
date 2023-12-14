import org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.global.opencv_video.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.Moments;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_video.*;
import static org.bytedeco.opencv.global.opencv_videoio.*;
import static org.bytedeco.opencv.helper.opencv_imgproc.*;

public class ObjectTracking {

    public static void main(String[] args) {
        VideoCapture videoCapture = new VideoCapture(0);

        double[] targetColor = {30, 150, 50}; // BGR color of the target object

        Mat targetColorHsv = new Mat();
        Mat frame = new Mat();
        Rect trackWindow = new Rect();

        while (true) {
            videoCapture.read(frame);

            if (frame.empty()) {
                System.out.println("End of video stream");
                break;
            }

            if (trackWindow.area() == 0) {
                targetColorHsv = getTargetColorInHsv(frame, targetColor);
                trackWindow = getBoundingRectangle(frame, targetColorHsv);
                meanShift(frame, trackWindow, new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1));
            } else {
                RotatedRect rotatedRect = meanShift(frame, trackWindow, new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1));

                if (trackWindow.area() > 100) {
                    ellipse(frame, rotatedRect, new Scalar(0, 0, 255), 2, LINE_AA, 0);
                } else {
                    trackWindow = new Rect();
                }
            }

            imshow("Object Tracking", frame);

            if (waitKey(30) >= 0) {
                break;
            }
        }

        videoCapture.release();
        cvDestroyAllWindows();
    }

    private static Mat getTargetColorInHsv(Mat frame, double[] targetColor) {
        Mat targetColorBgr = new Mat(1, 1, CV_8UC3, new Scalar(targetColor));
        Mat targetColorHsv = new Mat();
        cvtColor(targetColorBgr, targetColorHsv, COLOR_BGR2HSV);
        return targetColorHsv;
    }

    private static Rect getBoundingRectangle(Mat frame, Mat targetColorHsv) {
        Mat frameHsv = new Mat();
        cvtColor(frame, frameHsv, COLOR_BGR2HSV);

        Mat mask = new Mat();
        inRange(frameHsv, targetColorHsv, new Scalar(180, 255, 255), mask);

        Mat hist = new Mat();
        int histSize = 16;
        int[] channels = {0};
        int[] histSizeArray = {histSize};
        float[] range = {0, 180};
        float[][] ranges = {range};

        calcHist(frameHsv, 1, channels, mask, hist, 1, histSizeArray, ranges);

        normalize(hist, hist, 0, 255, NORM_MINMAX);

        Rect trackWindow;
        calcBackProject(frameHsv, 1, channels, hist, mask, hist, ranges, 255.0);

        CamShift(mask, new Rect(0, 0, frame.cols(), frame.rows()), new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1));
        minMaxLoc(mask, null, null, null, trackWindow, null);

        return trackWindow;
    }
}
