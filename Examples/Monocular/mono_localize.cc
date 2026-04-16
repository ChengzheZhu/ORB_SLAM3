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
 *   CameraTrajectory.txt   — TUM-format per-frame pose for localized frames
 *   KeyFrameTrajectory.txt — TUM-format keyframe poses
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
    // Atlas is loaded via System.LoadAtlasFromFile key in the YAML config.
    ORB_SLAM3::System SLAM(argv[1], argv[2],
                           ORB_SLAM3::System::MONOCULAR, bUseViewer);
    float imageScale = SLAM.GetImageScale();

    // Localization-only: do not extend the map
    cout << "Activating localization mode (map is read-only).\n";
    SLAM.ActivateLocalizationMode();

    // ── main loop ─────────────────────────────────────────────────────────
    int nImages = (int)vFilenames.size();
    double fakeInterval = 1.0 / 30.0;   // 30 fps simulated between repeats

    for (int ni = 0; ni < nImages; ni++)
    {
        cout << "\n[" << ni + 1 << "/" << nImages << "] "
             << vFilenames[ni] << "\n";

        cv::Mat im_orig = cv::imread(vFilenames[ni], cv::IMREAD_UNCHANGED);
        if (im_orig.empty())
        {
            cerr << "  WARNING: failed to read image, skipping.\n";
            continue;
        }

        if (imageScale != 1.f)
        {
            int w = (int)(im_orig.cols * imageScale);
            int h = (int)(im_orig.rows * imageScale);
            cv::resize(im_orig, im_orig, cv::Size(w, h));
        }

        // Feed the same image nRepeat times with incrementing timestamps.
        // This gives the relocalization thread multiple shots without motion.
        double t_base = vTimestamps[ni];
        for (int r = 0; r < nRepeat; r++)
        {
            double tframe = t_base + r * fakeInterval;

#ifdef COMPILEDWITHC11
            auto t1 = std::chrono::steady_clock::now();
#else
            auto t1 = std::chrono::monotonic_clock::now();
#endif
            SLAM.TrackMonocular(im_orig, tframe);
#ifdef COMPILEDWITHC11
            auto t2 = std::chrono::steady_clock::now();
#else
            auto t2 = std::chrono::monotonic_clock::now();
#endif
            double ttrack = std::chrono::duration_cast<
                std::chrono::duration<double>>(t2 - t1).count();

            // Pace playback so SLAM threads (local mapping, loop closing) keep up
            if (ttrack < fakeInterval)
                usleep((unsigned int)((fakeInterval - ttrack) * 1e6));
        }
    }

    SLAM.Shutdown();

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    cout << "\nTrajectory saved.\n";
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
