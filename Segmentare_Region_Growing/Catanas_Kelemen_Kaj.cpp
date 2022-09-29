// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <climits>
#include <cmath>
#include <math.h>
#include <random>

using namespace std;


////////////////////////////////////////////////PROIECT REGION GROWING - Catanas Kelemen Kaj/////////////////////////////////////////////////////////////////


bool isInside(int height, int width, int i, int j)
{
	if (0 <= i && i < height && 0 <= j && j < width)
	{
		return true;
	}
	return false;
}

Mat_<Vec3b> conv(Mat_ <Vec3b> src, Mat_<float> H) // FTJ
{
	int height = src.rows;
	int width = src.cols;
	Mat_<Vec3b> dst = src.clone();

	int w = H.cols;
	int k = (w - 1) / 2;

	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			double s2 = 0, s1 = 0, s0 = 0;
			for (int u = 0; u <= w - 1; u++)
			{
				for (int v = 0; v <= w - 1; v++)
				{
					s0 += H(u, v) * src.at<Vec3b>(i + u - k, j + v - k)[0];
					s1 += H(u, v) * src.at<Vec3b>(i + u - k, j + v - k)[1];
					s2 += H(u, v) * src.at<Vec3b>(i + u - k, j + v - k)[2];
				}
			}
			s2 /= 0.9983;
			s1 /= 0.9983;
			s0 /= 0.9983;
			dst.at<Vec3b>(i, j) = Vec3b(s0, s1, s2);
		}
	}
	return dst;
}


double dist(double x1, double y1, double x2, double y2) // Distanta Euclidiana
{
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}


void segmentare_RegionGrowing()
{

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double value;
		cout << "\nValue:"; cin >> value;

		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat_<Vec3b> gauss = src.clone();

		//1.)Filtrare

		float m1[] = { 1,  4,  7,  4,  1,
					   4, 16, 26, 16,  4,
					   7, 26, 41, 26,  7,
					   4, 16, 26, 16,  4,
					   1,  4,  7,  4,  1 }; //273

		float m2[] = { 0.0005, 0.0050, 0.0109, 0.0050, 0.0005,
					   0.0050, 0.0521, 0.1139, 0.0521, 0.0050,
					   0.0109, 0.1139, 0.2487, 0.1139, 0.0109,
					   0.0050, 0.0521, 0.1139, 0.0521, 0.0050,
					   0.0005, 0.0050, 0.0109, 0.0050, 0.0005 }; //0.9983

		// sigma=0.8 -> w = 6 * 0.8 = 4.8 ~ 5

		Mat_<float> h(5, 5, m2);

		gauss = conv(src, h); //FTJ

		//conversie
		Mat_<Vec3b> labeled;

		cvtColor(gauss, labeled, COLOR_BGR2Luv);


		//calcul deviatie
		double sum_u = 0, sum_v = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				sum_u = sum_u + labeled.at<Vec3b>(i, j)[1];
				sum_v = sum_v + labeled.at<Vec3b>(i, j)[2];
			}
		}

		double medie_u = (double)sum_u / (height * width);
		double medie_v = (double)sum_v / (height * width);

		double rez1 = 0, dev1 = 0, rez2 = 0, dev2 = 0;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dev1 = labeled.at<Vec3b>(i, j)[1] - medie_u;
				rez1 = rez1 + (dev1 * dev1);

				dev2 = labeled.at<Vec3b>(i, j)[2] - medie_v;
				rez2 = rez2 + (dev2 * dev2);
			}
		}
		double ch1_std = (double)sqrt(rez1 / (height * width)); //deviatie u
		double ch2_std = (double)sqrt(rez2 / (height * width)); //deviatie v


		double T = value * dist(0, 0, ch1_std, ch2_std);


		// 3.) Etichetare

		Mat labels = Mat::zeros(height, width, CV_8UC1);
		int label = 0, N;

		int di[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
		int dj[8] = { 1, 0, -1, -1, -1, 0, 1, 1 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<uchar>(i, j) == 0)
				{
					label++;

					queue<Point> Q;
					Q.push(Point(i, j));

					labels.at<uchar>(i, j) = label;

					N = 1;

					double ch0_avg = 0.0, ch1_avg = 0.0, ch2_avg = 0.0;
					int nr = 0;


					for (int k = 0; k < 8; k++)
					{
						if (isInside(height, width, i + di[k], j + dj[k]) && labels.at<uchar>(i + di[k], j + dj[k]) == 0)
						{
							ch0_avg = (nr * ch0_avg + labeled.at<Vec3b>(i + di[k], j + dj[k])[0]) / (nr + 1);
							ch1_avg = (nr * ch1_avg + labeled.at<Vec3b>(i + di[k], j + dj[k])[1]) / (nr + 1);
							ch2_avg = (nr * ch2_avg + labeled.at<Vec3b>(i + di[k], j + dj[k])[2]) / (nr + 1);
							nr++;
						}
					}

					do
					{
						Point q = Q.front();

						for (int k = 0; k < 8; k++)
						{
							if (isInside(height, width, q.x + di[k], q.y + dj[k]))
							{
								double ch0 = labeled.at<Vec3b>(q.x + di[k], q.y + dj[k])[0];
								double ch1 = labeled.at<Vec3b>(q.x + di[k], q.y + dj[k])[1];
								double ch2 = labeled.at<Vec3b>(q.x + di[k], q.y + dj[k])[2];

								double distanta = dist(ch1, ch2, ch1_avg, ch2_avg);

								if (labels.at<uchar>(q.x + di[k], q.y + dj[k]) == 0 && distanta < T)
								{
									Q.push(Point(q.x + di[k], q.y + dj[k]));
									labels.at<uchar>(q.x + di[k], q.y + dj[k]) = label;

									ch0_avg = (N * ch0_avg + ch0) / (N + 1);
									ch1_avg = (N * ch1_avg + ch1) / (N + 1);
									ch2_avg = (N * ch2_avg + ch2) / (N + 1);

									N++;
								}

							}
						}

						Q.pop();

					} while (!Q.empty());

					// colorare pixeli cu eticheta curenta
					for (int ii = 0; ii < height; ii++)
					{
						for (int jj = 0; jj < width; jj++)
						{
							if (labels.at<uchar>(ii, jj) == label)
								labeled.at<Vec3b>(ii, jj) = Vec3b(ch0_avg, ch1_avg, ch2_avg);
						}
					}

				}
			}
		}

		Mat post = labeled.clone();

		Mat_<Vec3b> lab;
		cvtColor(labeled, lab, COLOR_Luv2BGR);


		//4.) Eroziune & Dilatare

		int done = 0;


		for (int l = 0; l < 2; l++) //3 eroziuni
		{
			Mat new_labels = labels.clone();

			for (int i = 2; i < height - 2; i++)
			{
				for (int j = 2; j < width - 2; j++)
				{
					if (labels.at<uchar>(i, j) != 0)
					{
						for (int k = 0; k < 8; k++)
						{
							if (labels.at<uchar>(i + di[k], j + dj[k]) != labels.at<uchar>(i, j))
							{
								new_labels.at<uchar>(i, j) = 0;
							}
						}
					}
				}
			}
			labels = new_labels;
		}


		do // n dilatari
		{
			done = 0;

			Mat new_labels = labels.clone();

			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (labels.at<uchar>(i, j) != 0)
					{
						for (int k = 0; k < 8; k++)
						{
							if (labels.at<uchar>(i + di[k], j + dj[k]) == 0)
								new_labels.at<uchar>(i + di[k], j + dj[k]) = labels.at<uchar>(i, j);

						}
					}
					else
					{
						for (int k = 0; k < 8; k++)
						{
							if (labels.at<uchar>(i + di[k], j + dj[k]) != 0)
							{
								done = 1;
							}
						}
					}
				}
			}

			labels = new_labels;

		} while (done != 0);

		int max = INT_MIN;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<uchar>(i, j) > max) max = labels.at<uchar>(i, j); // eticheta maxima
			}
		}


		for (int e = 1; e <= max; e++)
		{
			double s0 = 0, s1 = 0, s2 = 0, nr = 0;

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (labels.at<uchar>(i, j) == e)
					{
						s0 += labeled.at<Vec3b>(i, j)[0];
						s1 += labeled.at<Vec3b>(i, j)[1];
						s2 += labeled.at<Vec3b>(i, j)[2];
						nr++;
					}

				}
			}

			if (nr != 0)
			{
				s0 /= nr;
				s1 /= nr;
				s2 /= nr;
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (labels.at<uchar>(i, j) == e)
					{
						post.at<Vec3b>(i, j) = Vec3b(s0, s1, s2);
					}

				}
			}

		}

		Mat_<Vec3b> fin;
		cvtColor(post, fin, COLOR_Luv2BGR);

		imshow("Original", src);
		imshow("Filtered", gauss);
		imshow("Labeled", lab);
		imshow("Final", fin);

		waitKey(0);

	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");

		//Proiect
		printf(" 1 - Segmentare Region Growing\n");

		//EXIT
		printf(" 0 - Exit\n\n");
		printf("Option: ");

		scanf("%d", &op);
		switch (op)
		{

		case 1:
			segmentare_RegionGrowing();
			break;
		default:
			destroyAllWindows();

		}
	} while (op != 0);
	return 0;
}