#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(196) void _occa_cubatureGeometricFactorsHex3D_0(const int Nelements,
                                                                                                                      const double * __restrict__ D,
                                                                                                                      const double * __restrict__ x,
                                                                                                                      const double * __restrict__ y,
                                                                                                                      const double * __restrict__ z,
                                                                                                                      const double * __restrict__ cubInterpT,
                                                                                                                      const double * __restrict__ cubW,
                                                                                                                      double * __restrict__ cubvgeo) {
  {
    int element = 0 + blockIdx.x;
    __shared__ double s_cubInterpT[10][14];
    __shared__ double s_cubw[14];
    __shared__ double s_D[10][10];
    __shared__ double s_x[10][10];
    __shared__ double s_y[10][10];
    __shared__ double s_z[10][10];
    __shared__ double s_cubxre[10][14];
    __shared__ double s_cubxse[10][14];
    __shared__ double s_cubxte[10][14];
    double r_x[10], r_y[10], r_z[10];
    __shared__ double s_xre[10][10];
    __shared__ double s_xse[10][10];
    __shared__ double s_xte[10][10];
    __shared__ double s_yre[10][10];
    __shared__ double s_yse[10][10];
    __shared__ double s_yte[10][10];
    __shared__ double s_zre[10][10];
    __shared__ double s_zse[10][10];
    __shared__ double s_zte[10][10];
    __shared__ double s_cubyre[10][14];
    __shared__ double s_cubyse[10][14];
    __shared__ double s_cubyte[10][14];
    __shared__ double s_cubzre[10][14];
    __shared__ double s_cubzse[10][14];
    __shared__ double s_cubzte[10][14];

    // TODO: reduce register pressure
    double r_cubxre[14];
    double r_cubxse[14];
    double r_cubxte[14];
    double r_cubyre[14];
    double r_cubyse[14];
    double r_cubyte[14];
    double r_cubzre[14];
    double r_cubzse[14];
    double r_cubzte[14];
    {
      int j = 0 + threadIdx.y;
      {
        int i = 0 + threadIdx.x;
        const int id = i + j * 14;
        if (j == 0) {
          s_cubw[i] = cubW[i];
        }
        if (id < 10 * 14) {
          s_cubInterpT[j][i] = cubInterpT[id];
        }
        if (i < 10 && j < 10) {
          s_D[j][i] = D[j * 10 + i];
        }
        for (int k = 0; k < 14; ++k) {
          r_cubxre[k] = 0;
          r_cubxse[k] = 0;
          r_cubxte[k] = 0;
          r_cubyre[k] = 0;
          r_cubyse[k] = 0;
          r_cubyte[k] = 0;
          r_cubzre[k] = 0;
          r_cubzse[k] = 0;
          r_cubzte[k] = 0;
        }
      }
    }
    __syncthreads();
    for (int k = 0; k < 10; ++k) {
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          if (i < 10 && j < 10) {
            const int id = element * 1000 + k * 10 * 10 + j * 10 + i;
            s_x[j][i] = x[id];
            s_y[j][i] = y[id];
            s_z[j][i] = z[id];
            if (k == 0) {
              for (int l = 0; l < 10; ++l) {
                const int other_id = element * 1000 + l * 10 * 10 + j * 10 + i;
                r_x[l] = x[other_id];
                r_y[l] = y[other_id];
                r_z[l] = z[other_id];
              }
            }
          }
        }
      }
      ;
      __syncthreads();
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          if (i < 10 && j < 10) {
            double xr = 0, yr = 0, zr = 0;
            double xs = 0, ys = 0, zs = 0;
            double xt = 0, yt = 0, zt = 0;
            for (int m = 0; m < 10; ++m) {
              const double Dim = s_D[i][m];
              const double Djm = s_D[j][m];
              const double Dkm = s_D[k][m];
              xr += Dim * s_x[j][m];
              xs += Djm * s_x[m][i];
              xt += Dkm * r_x[m];
              yr += Dim * s_y[j][m];
              ys += Djm * s_y[m][i];
              yt += Dkm * r_y[m];
              zr += Dim * s_z[j][m];
              zs += Djm * s_z[m][i];
              zt += Dkm * r_z[m];
            }
            // store results in shmem array
            s_xre[j][i] = xr;
            s_xse[j][i] = xs;
            s_xte[j][i] = xt;
            s_yre[j][i] = yr;
            s_yse[j][i] = ys;
            s_yte[j][i] = yt;
            s_zre[j][i] = zr;
            s_zse[j][i] = zs;
            s_zte[j][i] = zt;
          }
        }
      }
      ;
      __syncthreads();
      {
        int b = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          if (b < 10) {
            double xr1 = 0, xs1 = 0, xt1 = 0;
            double yr1 = 0, ys1 = 0, yt1 = 0;
            double zr1 = 0, zs1 = 0, zt1 = 0;
            for (int a = 0; a < 10; ++a) {
              double Iia = s_cubInterpT[a][i];
              xr1 += Iia * s_xre[b][a];
              xs1 += Iia * s_xse[b][a];
              xt1 += Iia * s_xte[b][a];
              yr1 += Iia * s_yre[b][a];
              ys1 += Iia * s_yse[b][a];
              yt1 += Iia * s_yte[b][a];
              zr1 += Iia * s_zre[b][a];
              zs1 += Iia * s_zse[b][a];
              zt1 += Iia * s_zte[b][a];
            }
            s_cubxre[b][i] = xr1;
            s_cubxse[b][i] = xs1;
            s_cubxte[b][i] = xt1;
            s_cubyre[b][i] = yr1;
            s_cubyse[b][i] = ys1;
            s_cubyte[b][i] = yt1;
            s_cubzre[b][i] = zr1;
            s_cubzse[b][i] = zs1;
            s_cubzte[b][i] = zt1;
          }
        }
      }
      ;
      __syncthreads();

      // interpolate in 's'
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          double xr2 = 0, xs2 = 0, xt2 = 0;
          double yr2 = 0, ys2 = 0, yt2 = 0;
          double zr2 = 0, zs2 = 0, zt2 = 0;
          // interpolate in b
          for (int b = 0; b < 10; ++b) {
            double Ijb = s_cubInterpT[b][j];
            xr2 += Ijb * s_cubxre[b][i];
            xs2 += Ijb * s_cubxse[b][i];
            xt2 += Ijb * s_cubxte[b][i];
            yr2 += Ijb * s_cubyre[b][i];
            ys2 += Ijb * s_cubyse[b][i];
            yt2 += Ijb * s_cubyte[b][i];
            zr2 += Ijb * s_cubzre[b][i];
            zs2 += Ijb * s_cubzse[b][i];
            zt2 += Ijb * s_cubzte[b][i];
          }

          // interpolate in k progressively
          for (int c = 0; c < 14; ++c) {
            double Ick = s_cubInterpT[k][c];
            r_cubxre[c] += Ick * xr2;
            r_cubxse[c] += Ick * xs2;
            r_cubxte[c] += Ick * xt2;
            r_cubyre[c] += Ick * yr2;
            r_cubyse[c] += Ick * ys2;
            r_cubyte[c] += Ick * yt2;
            r_cubzre[c] += Ick * zr2;
            r_cubzse[c] += Ick * zs2;
            r_cubzte[c] += Ick * zt2;
          }
        }
      }
      ;
      __syncthreads();
    }
    for (int k = 0; k < 14; ++k) {
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          const double xr = r_cubxre[k], xs = r_cubxse[k], xt = r_cubxte[k];
          const double yr = r_cubyre[k], ys = r_cubyse[k], yt = r_cubyte[k];
          const double zr = r_cubzre[k], zs = r_cubzse[k], zt = r_cubzte[k];
          const double J = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt);
          const double invJ = 1.0 / J;
          const double JW = J * s_cubw[i] * s_cubw[j] * s_cubw[k];
          const double drdx = (ys * zt - zs * yt) * invJ;
          const double drdy = -(xs * zt - zs * xt) * invJ;
          const double drdz = (xs * yt - ys * xt) * invJ;
          const double dsdx = -(yr * zt - zr * yt) * invJ;
          const double dsdy = (xr * zt - zr * xt) * invJ;
          const double dsdz = -(xr * yt - yr * xt) * invJ;
          const double dtdx = (yr * zs - zr * ys) * invJ;
          const double dtdy = -(xr * zs - zr * xs) * invJ;
          const double dtdz = (xr * ys - yr * xs) * invJ;
          const int gid = element * 2744 * 12 + k * 14 * 14 + j * 14 + i;
          cubvgeo[gid + 0 * 2744] = drdx;
          cubvgeo[gid + 1 * 2744] = drdy;
          cubvgeo[gid + 2 * 2744] = drdz;
          cubvgeo[gid + 3 * 2744] = dsdx;
          cubvgeo[gid + 4 * 2744] = dsdy;
          cubvgeo[gid + 5 * 2744] = dsdz;
          cubvgeo[gid + 6 * 2744] = dtdx;
          cubvgeo[gid + 7 * 2744] = dtdy;
          cubvgeo[gid + 8 * 2744] = dtdz;
          cubvgeo[gid + 9 * 2744] = J;
          cubvgeo[gid + 10 * 2744] = JW;
          cubvgeo[gid + 11 * 2744] = 1.0 / JW;
        }
      }
      ;
    }
  }
}

