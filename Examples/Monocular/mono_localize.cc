/**
 * mono_localize.cc — Monocular localization against a pre-built ORB-SLAM3 Atlas.
 *
 * Loads a saved map (via System.LoadAtlasFromFile in the YAML config), activates
 * localization mode (no new map points created), then feeds each input image through
 * TrackMonocular.  Each image is repeated --repeat times to give the relocalization
 * loop multiple attempts from the same viewpoint.
 *
 * Usage:
 *   ./mono_localize  path_to_vocabulary  path_to_settings  path_to_image_list
 *                    [repeat=5]  [viewer=0]
 *
 * path_to_image_list: plain text file, one entry per line:
 *   <timestamp> <absolute_or_relative_image_path>
 *   e.g.
 *     0.000 /home/user/photos/DSC07731.JPG
 *     1.000 /home/user/photos/DSC07732.JPG
 *   Lines starting with # are ignored.
 *
 * Output (written to CWD):
 *   LocalizedPoses.txt     — per-image Twc pose for successfully localized frames
 *                            format: timestamp image_path tx ty tz qx qy qz qw
 *   KeyFrameTrajectory.txt — TUM-format keyframe poses (original map KFs)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <System.h>

using namespace std;

void LoadImageList(const string &path, vector<string> &filenames,
                   vector<double> &timestamps);

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 6)
    {
        cerr << "\nUsage: ./mono_localize  vocab  settings  image_list"
                "  [repeat=5]  [viewer=0]\n";
        return 1;
    }

    int    nRepeat    = (argc >= 5) ? stoi(argv[4]) : 5;
    bool   bUseViewer = !(argc >= 6 && string(argv[5]) == "0");

    // ── load image list ───────────────────────────────────────────────────
    vector<string> vFilenames;
    vector<double> vTimestamps;
    LoadImageList(string(argv[3]), vFilenames, vTimestamps);

    if (vFilenames.empty())
    {
        cerr << "ERROR: no images loaded from " << argv[3] << "\n";
        return 1;
    }
    cout << "Images to localize: " << vFilenames.size()
         << "  (each repeated " << nRepeat << "x)\n";

    // ── create SLAM system ────────────────────────────────────────────────
    ORB_SLAM3::System SLAM(argv[1], argv[2],
                           ORB_SLAM3::System::MONOCULAR, bUseViewer);
    float imageScale = SLAM.GetImageScale();

    cout << "Activating localization mode (map is read-only).\n";
    SLAM.ActivateLocalizationMode();

    // ── main loop ─────────────────────────────────────────────────────────
    int nImages = (int)vFilenames.size();
    double fakeInterval = 1.0 / 30.0;

    // Store per-image result: (timestamp, path, Twc, localized)
    struct Result { double ts; string path; Sophus::SE3f Twc; bool ok; };
    vector<Result> vResults;

    // tCurrent strictly increases across ALL repeats of ALL images.
    // Seed it from the first image's timestamp so we stay above atlas KF range.
    double tCurrent = vTimestamps.empty() ? 0.0 : vTimestamps[0];

    for (int ni = 0; ni < nImages; ni++)
    {
        cout << "\n[" << ni + 1 << "/" << nImages << "] "
             << vFilenames[ni] << "\n";

        cv::Mat im_orig = cv::imread(vFilenames[ni], cv::IMREAD_UNCHANGED);
        if (im_orig.empty())
        {
            cerr << "  WARNING: failed to read image, skipping.\n";
            // Advance tCurrent past this image's slot
            tCurrent += nRepeat * fakeInterval;
            vResults.push_back({vTimestamps[ni], vFilenames[ni],
                                Sophus::SE3f(), false});
            continue;
        }

        if (imageScale != 1.f)
        {
            int w = (int)(im_orig.cols * imageScale);
            int h = (int)(im_orig.rows * imageScale);
            cv::resize(im_orig, im_orig, cv::Size(w, h));
        }

        // Feed the same image nRepeat times with strictly increasing timestamps.
        // Capture the pose from the last successful (state==OK) repeat.
        Sophus::SE3f Tcw;
        bool localized = false;

        for (int r = 0; r < nRepeat; r++)
        {
            double tframe = tCurrent;
            tCurrent += fakeInterval;

#ifdef COMPILEDWITHC11
            auto t1 = std::chrono::steady_clock::now();
#else
            auto t1 = std::chrono::monotonic_clock::now();
#endif
            Sophus::SE3f TcwCur = SLAM.TrackMonocular(im_orig, tframe);
            int state = SLAM.GetTrackingState();
            if (state == 2)   // Tracking::OK
            {
                localized = true;
                Tcw = TcwCur;   // keep last successful pose
            }
#ifdef COMPILEDWITHC11
            auto t2 = std::chrono::steady_clock::now();
#else
            auto t2 = std::chrono::monotonic_clock::now();
#endif
            double ttrack = std::chrono::duration_cast<
                std::chrono::duration<double>>(t2 - t1).count();

            if (ttrack < fakeInterval)
                usleep((unsigned int)((fakeInterval - ttrack) * 1e6));
        }

        // Twc = inverse of Tcw  (camera position in world frame)
        Sophus::SE3f Twc = Tcw.inverse();
        // Extra pause after last repeat: let the reloc thread catch up
        usleep(300000);  // 300ms

        vResults.push_back({vTimestamps[ni], vFilenames[ni], Twc, localized});

        if (localized)
            cout << "  ✓ localized\n";
        else
            cout << "  ✗ not localized\n";
    }

    // ── write LocalizedPoses.txt before Shutdown() to survive cleanup crash ──
    // Format: timestamp image_path tx ty tz qx qy qz qw
    {
        ofstream f("LocalizedPoses.txt");
        f << fixed << setprecision(9);
        f << "# timestamp image_path tx ty tz qx qy qz qw\n";
        int nOk = 0;
        for (auto &r : vResults)
        {
            if (!r.ok) continue;
            Eigen::Vector3f    t = r.Twc.translation();
            Eigen::Quaternionf q = r.Twc.unit_quaternion();
            f << r.ts << " " << r.path << " "
              << t.x() << " " << t.y() << " " << t.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << "\n";
            nOk++;
        }
        cout << "\nLocalized " << nOk << "/" << nImages
             << " images → LocalizedPoses.txt\n";
    }

    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectory saved.\n";

    SLAM.Shutdown();
    return 0;
}

void LoadImageList(const string &path, vector<string> &filenames,
                   vector<double> &timestamps)
{
    ifstream f(path);
    if (!f.is_open())
    {
        cerr << "ERROR: cannot open image list: " << path << "\n";
        return;
    }
    string line;
    while (getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        double t;
        string fname;
        if (!(ss >> t >> fname)) continue;
        timestamps.push_back(t);
        filenames.push_back(fname);
    }
}
