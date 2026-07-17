/*
 * ParFE: a micro-FE solver for trabecular bone modeling
 * Copyright (C) 2006, Uche Mennel and Marzio Sala, Cyril Flaig
 *
 * ParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */

/* Taken from ParFE */

/* Documentation see Programming the Finite Element Method, 3rd Edition I. M.
 * Smith, D. V. Griffith
 */


//#define SELFTEST   //compile the main
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <vector>

using Eigen::MatrixXd;
using std::complex;
//this code is fortranreplacment -> colmojor
//#define rowmajor

/*#ifdef romajor
#define ACCESS(r,c,rows,cols)\
    (r)*(cols) + (c)
#else
*/
#define ACCESS(r,c,rows,cols)\
    (c)*(rows) + (r)
//#endif


int  inline matmulAB(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc);
int  inline matmulABt(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc);
int  inline matmulAtB(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc);

void inline d_mat(double *,double, double, int);
void inline sample(char, const int, double *, double *);
void inline shape_der(double *, double *, int, int, int, int);
double inline determinant(double*,int n);
void inline invert(double *, double *, int n);
void inline b_mat(double *, double *,int nod, int ndof, int nst);
void inline aXpY(double, double *, double *,int mode, int dim); //if mode == 1 y=ax+y esle y=ax
void invar(double *stress, double &sigma, double &dsbar, double &theta, int nst);
//void invar(double *stress, double &sigma, double &dsbar, double &theta, int nst);

void Stiffness_Matrix(const double* mat_prop, const int& nprops, const int& nod, const int& ndof, const int& ndim, const int& nip, const int& nst, const double* g_num, double* km)
{
    // mat_prop[nprops],g_num[ndim,nod], km(ndof,ndof)
    char element = 'h';
    int i; //oldvar error

    double det,e,v;

    //arrays
    double *points  = new double[nip * ndim];
    double *D   = new double[nst * nst];
    double *weights = new double[nip];
    double *der   = new double[ndim * nod];
    double *jac     = new double[ndim * ndim];
    double *deriv = new double[ndim * nod];
    double *B     = new double[nst * ndof];
    double *coord = new double[nod * ndim];
    double *BtD     = new double[ndof * nst];
    double *BtDB  = new double [ndof * ndof];
    double *jac1     = new double[ndim * ndim];


    e = mat_prop[0];
    v = mat_prop[1];
    d_mat(D,e,v, nst);
    sample(element,nip, points, weights);
    //set km to zero km = 0.0;
    for (i=0; i < ndof*ndof; i++)
        km[i]=0.0;


//     cout << endl;
//     for (int j=0;j<ndim;j++) {
//         for (i=0;i<nip+9;i++)
//             cout << setiosflags(ios::scientific) << setw(22) << setprecision(14) << points[ACCESS(i,j,nip,ndim)] << " ";
//         cout << endl;
//     }
//     cout << "weights \n";

    for (i=0; i < nip; i++) {
        shape_der (der,points,i,nod,ndim,nip); //commputes the deratives on the int points acoording to local coords
        matmulABt(der,ndim,nod, g_num,ndim,nod, jac1,ndim,ndim); //compute the jacobian
        det = determinant(jac1, ndim);
        invert(jac1,jac,ndim);
        matmulAB(jac,ndim,ndim, der,ndim,nod, deriv,ndim,nod); //computes J^1AS
        b_mat(B,deriv,nod, ndof,nst);
        matmulAtB(B,nst,ndof, D,nst,nst, BtD,ndof,nst);
        matmulAB(BtD,ndof,nst,B,nst,ndof,BtDB,ndof,ndof);
        aXpY(det*weights[i],BtDB,km,1,ndof * ndof); // 1 means km+= .....
    }
     delete[] points;
     delete[] D;
     delete[] weights;
     delete[] der;
     delete[] jac;
     delete[] deriv;
     delete[] B;
     delete[] coord;
     delete[] BtD;
     delete[] BtDB;
	 delete[] jac1;
}

void Initial_Strain_Load(const double* mat_prop, const int& nprops, const int& nod, const int& ndof, const int& ndim, const int& nip, const int& nst, const double* g_num, const double* plastic_strain, double* load)
{
    char element = 'h';
    int i;
    double det, e, v;

    double *points  = new double[nip * ndim];
    double *D   = new double[nst * nst];
    double *weights = new double[nip];
    double *der   = new double[ndim * nod];
    double *jac     = new double[ndim * ndim];
    double *deriv = new double[ndim * nod];
    double *B     = new double[nst * ndof];
    double *stress = new double[nst];
    double *gauss_load = new double[ndof];
    double *jac1     = new double[ndim * ndim];

    e = mat_prop[0];
    v = mat_prop[1];
    d_mat(D, e, v, nst);
    sample(element, nip, points, weights);
    for (i = 0; i < ndof; i++)
        load[i] = 0.0;

    for (i = 0; i < nip; i++) {
        shape_der(der, points, i, nod, ndim, nip);
        matmulABt(der, ndim, nod, g_num, ndim, nod, jac1, ndim, ndim);
        det = determinant(jac1, ndim);
        invert(jac1, jac, ndim);
        matmulAB(jac, ndim, ndim, der, ndim, nod, deriv, ndim, nod);
        b_mat(B, deriv, nod, ndof, nst);
        matmulAB(D, nst, nst, plastic_strain + i * nst, nst, 1, stress, nst, 1);
        matmulAtB(B, nst, ndof, stress, nst, 1, gauss_load, ndof, 1);
        aXpY(det * weights[i], gauss_load, load, 1, ndof);
    }

    delete[] points;
    delete[] D;
    delete[] weights;
    delete[] der;
    delete[] jac;
    delete[] deriv;
    delete[] B;
    delete[] stress;
    delete[] gauss_load;
    delete[] jac1;
}
using Eigen::EigenSolver;
void Element_Stress(const double* mat_prop, const int& nprops, const int& nod, const int& ndof, const int& ndim, const int& nip, 
                    const int& nst, const double* g_num, double* eld, double* strain, double* stress, double* e_vol, double* e_dev, 
                    double* sigma, double* theta, double* e_xx,double* e_yy,double* e_zz,double* e_xy,double* e_yz,double* e_xz, 
                    double* s_xx,double* s_yy,double* s_zz,double* s_xy,double* s_yz,double* s_xz,double* DP_s,double* DP_e, 
                    double* e1_out, double* e2_out, double* e3_out, double* s1_out, double* s2_out, double* s3_out,
                    bool compute_von_mises, bool compute_component_fields, bool compute_dp, bool compute_principal) {
    int i,j,n; //oldvar: error
    double e,v;
    char element;
    double *points  = new double[nip * ndim],   *D   = new double[nst * nst];
    double *weights = new double[nip],         *der   = new double[ndim * nod];
    double *jac     = new double[ndim * ndim],  *deriv = new double[ndim * nod];
    double *B     = new double[nst * ndof],   *coord = new double[nod * ndim];
    double *temp    = new double[ndof],          *Bld = new double[nst];
    double *deeld = new double[nst],           *jac1     = new double[ndim * ndim];
    double e11, e22, e33, e12, e13, e23, s11, s22, s33, s12, s13, s23, e1, e2, e3, s1, s2, s3;

    element = 'h';
    e = mat_prop[0];
    v = mat_prop[1];
    for (i=0; i < (nst+1) *nip; i++) {
        strain[i] = 0.0;
        stress[i] = 0.0;
    }


    d_mat(D,e,v,nst);
    sample(element, nip, points, weights);
    for (i=0; i < nip; i++) {
	    shape_der (der,points,i,nod,ndim,nip);
        matmulABt(der,ndim,nod, g_num,ndim,nod, jac1,ndim,ndim);
        //det = determinant(jac1, ndim);
        invert(jac1,jac,ndim);
        matmulAB(jac,ndim,ndim, der,ndim,nod, deriv,ndim,nod);
        b_mat(B,deriv,nod, ndof,nst);
        matmulAB(B,nst,ndof,eld,ndof,1,Bld,nst,1);
        for (n=0;n<nst;n++)
            strain[ACCESS(n,i,nst+1,nip)]=Bld[n];
        matmulAB(D,nst,nst,Bld,nst,1,deeld,nst,1);
        for (n=0;n<nst;n++)
            stress[ACCESS(n,i,nst+1,nip)]=deeld[n];

        s11 = stress[ACCESS(0,i,nst+1,nip)]; s22 = stress[ACCESS(1,i,nst+1,nip)]; s33 = stress[ACCESS(2,i,nst+1,nip)];
        s12 = stress[ACCESS(3,i,nst+1,nip)]; s23 = stress[ACCESS(4,i,nst+1,nip)]; s13 = stress[ACCESS(5,i,nst+1,nip)];
        e11 = strain[ACCESS(0,i,nst+1,nip)]; e22 = strain[ACCESS(1,i,nst+1,nip)]; e33 = strain[ACCESS(2,i,nst+1,nip)]; 
	    e12 = strain[ACCESS(3,i,nst+1,nip)]; e13 = strain[ACCESS(4,i,nst+1,nip)]; e23 = strain[ACCESS(5,i,nst+1,nip)];
	
        if (compute_principal) {
            MatrixXd s_tens(3,3);
	        s_tens(0,0) = s11; s_tens(1,1) = s22; s_tens(2,2)=s33;
	        s_tens(0,1)=s_tens(1,0) = s12; s_tens(0,2)=s_tens(2,0) = s13; s_tens(2,1)=s_tens(1,2) = s23;
	        EigenSolver<MatrixXd> es_s(s_tens);

	        complex<double> lambda_s = es_s.eigenvalues()[0];
            s1 = (double)lambda_s.real();
	        lambda_s = es_s.eigenvalues()[1];
	        s2 = (double)lambda_s.real();
	        lambda_s = es_s.eigenvalues()[2];
            s3 = (double)lambda_s.real();
            std::vector<double> s_temp = {s1, s2, s3};
            std::sort(s_temp.begin(), s_temp.end(), std::greater<double>());
            s1_out[0] = s_temp[0];
            s2_out[0] = s_temp[1];
            s3_out[0] = s_temp[2];

            MatrixXd e_tens(3,3);
            e_tens(0,0) = e11; e_tens(1,1) = e22; e_tens(2,2)=e33;
            e_tens(0,1)=e_tens(1,0) = e12; e_tens(0,2)=e_tens(2,0) = e13; e_tens(2,1)=e_tens(1,2) = e23;
            EigenSolver<MatrixXd> es_e(e_tens);

            complex<double> lambda_e = es_e.eigenvalues()[0];
            e1 =(double)lambda_e.real();
	        lambda_e = es_e.eigenvalues()[1];
            e2 =(double)lambda_e.real();
            lambda_e = es_e.eigenvalues()[2];
            e3 =(double)lambda_e.real();
            std::vector<double> e_temp = {e1, e2, e3};
            std::sort(e_temp.begin(), e_temp.end(), std::greater<double>());
            e1_out[0] = e_temp[0];
            e2_out[0] = e_temp[1];
            e3_out[0] = e_temp[2];

            e_vol[0] = (e1+e2+e3);
            e_dev[0] = sqrt(0.5*(((e1-e2)*(e1-e2)+(e2-e3)*(e2-e3)+(e1-e3)*(e1-e3))));
        }

        if (compute_component_fields) {
	        s_xx[0] = s11; s_yy[0] = s22; s_zz[0] = s33; s_xy[0] = s12; s_yz[0] = s23; s_xz[0] = s13;
	        e_xx[0] = e11; e_yy[0] = e22; e_zz[0] = e33; e_xy[0] = e12; e_yz[0] = e23; e_xz[0] = e13;
        }

        if (compute_dp) {
	        DP_s[0] = 0.02673*(s11+s22+s33) + sqrt((((s11-s22)*(s11-s22)+(s22-s33)*(s22-s33)+(s33-s11)*(s33-s11))/6.0)+(s12*s12)+(s13*s13)+(s23*s23));
	        DP_e[0] = 0.02673*(e11+e22+e33) + sqrt((((e11-e22)*(e11-e22)+(e22-e33)*(e22-e33)+(e33-e11)*(e33-e11))/6.0)+(e12*e12)+(e13*e13)+(e23*e23));
        }

        if (compute_von_mises) {
            invar(deeld, sigma[i], stress[ACCESS(nst,i,nst+1,nip)], theta[i],nst);
        }

        for(j=0;j<nst;j++)
            strain[ACCESS(nst,i,nst+1,nip)] = strain[ACCESS(nst,i,nst+1,nip)] + strain[ACCESS(j,i,nst+1,nip)]*stress[ACCESS(j,i,nst+1,nip)];
        strain[ACCESS(nst,i,nst+1,nip)]*=0.5;

    }

    delete[] points;
    delete[] D;
    delete[] weights;
    delete[] der;
    delete[] jac;
    delete[] deriv;
    delete[] B;
    delete[] coord;
    delete[] temp;
    delete[] Bld;
    delete[] deeld;
    delete[] jac1;

}

int  inline matmulAB(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc){
    double sum;
    int i,j,k;
//     if ((Ac != Br) || (Ar != Cr) ||(Bc != Cc)) {
//         cout << "*********************ERROR DIMENSION OF THE MATRIX NOT OKEY**************** \n";
//         return (-1);
//     }

    for(i=0; i<Ar; i++)
        for(j=0; j<Bc; j++) {
        sum=0;
        for(k=0; k<Br; k++)
            sum += A[ACCESS(i,k,Ar,Ac)]*B[ACCESS(k,j,Br,Bc)];
        C[ACCESS(i,j,Cr,Cc)] = sum;
        }

        return (0);
}

//matmult with the second matrix transposed
int  inline matmulABt(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc){
    double sum;
    int i,j,k;
//     if ((Ac != Bc) || (Ar != Cr) ||(Br != Cc)) {
//         cout << "*********************ERROR DIMENSION OF THE MATRIX NOT OKEY**************** \n";
//         return (-1);
//     }

    for(i=0; i<Ar; i++)
        for(j=0; j<Br; j++) {
        sum=0;
        for(k=0; k<Bc; k++)
            sum += A[ACCESS(i,k,Ar,Ac)]*B[ACCESS(j,k,Br,Bc)];
        C[ACCESS(i,j,Cr,Cc)] = sum;
        }

        return (0);
}

int  inline matmulAtB(const double *A, int Ar, int Ac, const double *B, int Br, int Bc, double *C, int Cr, int Cc){
    double sum;
    int i,j,k;
//     if ((Ar != Br) || (Ac != Cr) ||(Bc != Cc)) {
//         cout << "*********************ERROR DIMENSION OF THE MATRIX NOT OKEY**************** \n";
//         return (-1);
//     }

    for(i=0; i<Ac; i++)
        for(j=0; j<Bc; j++) {
        sum=0;
        for(k=0; k<Br; k++)
            sum += A[ACCESS(k,i,Ar,Ac)]*B[ACCESS(k,j,Br,Bc)];
        C[ACCESS(i,j,Cr,Cc)] = sum;
        }

        return (0);
}

void inline d_mat(double *d ,double e, double v, int nst){
    //double v1;
    double v2,c,vv;
    int i;
    //Write zeros in d
    for (i=0; i< nst*nst; i++)
        d[i]=0.0;

    //v1 = 1.0 - v;
    v2 = v/(1.0-v);
    vv = (1.0-2.0*v)/(1.0-v)*0.5;
    //set diag 1-3 to 1
    for (i =0; i<3;i++)
        d[ACCESS(i,i,nst,nst)] = 1.0;
    //set diag 4-6 to (1-2v)/(2(1-v))
    for (i =3; i<6;i++)
            d[ACCESS(i,i,nst,nst)] = vv;

    d[ACCESS(0,1,nst,nst)] = v2;
    d[ACCESS(1,0,nst,nst)] = v2;
    d[ACCESS(0,2,nst,nst)] = v2;
    d[ACCESS(2,0,nst,nst)] = v2;
    d[ACCESS(1,2,nst,nst)] = v2;
    d[ACCESS(2,1,nst,nst)] = v2;

    c = e/(2.0*(1.0+v)*vv);
    for (i=0; i< nst*nst; i++)
        d[i]*=c;

    return;
}

//TODO: not all 3 dimension are set. why?
void inline sample(char ch, const int nip, double *s, double *wt){
    //only hexahedron
    double b=0,c=0,root3=0,r15=0;
    double w[3],v[9];
    int i;

    //root3 = 1.0/sqrt(3.0);
    root3=0.57735026918962576452;
    //r15 = 0.2*sqrt(15.0);
    r15= 0.77459666924148337704;
    w[0]=5.0/9.0;
    w[1]=8.0/9.0;
    w[2]=5.0/9.0;
    for (int j= 0; j <3; j++) {
        for (i =0; i<3; i++) {
            v[3*j +i]=w[j]*w[i];
        }
    }


    if ('h' == ch) {
        switch (nip)
        {
            case 1:
                s[0]= 0.0;
                s[1]= 0.0;
                s[2]=0.0;
                wt[0] = 8.0;
                break;
            case 8:
                s[ACCESS(0,0,nip,3)] = root3;
                s[ACCESS(0,1,nip,3)] = root3;
                s[ACCESS(0,2,nip,3)] = root3;
                s[ACCESS(1,0,nip,3)] = root3;
                s[ACCESS(1,1,nip,3)] = root3;
                s[ACCESS(1,2,nip,3)] = -root3;
                s[ACCESS(2,0,nip,3)] = root3;
                s[ACCESS(2,1,nip,3)] = -root3;
                s[ACCESS(2,2,nip,3)] = root3;
                s[ACCESS(3,0,nip,3)] = root3;
                s[ACCESS(3,1,nip,3)] = -root3;
                s[ACCESS(3,2,nip,3)] = -root3;
                s[ACCESS(4,0,nip,3)] = -root3;
                s[ACCESS(4,1,nip,3)] = root3;
                s[ACCESS(4,2,nip,3)] = root3;
                s[ACCESS(5,0,nip,3)] = -root3;
                s[ACCESS(5,1,nip,3)] = -root3;
                s[ACCESS(5,2,nip,3)] = root3;
                s[ACCESS(6,0,nip,3)] = -root3;
                s[ACCESS(6,1,nip,3)] = root3;
                s[ACCESS(6,2,nip,3)] = -root3;
                s[ACCESS(7,0,nip,3)] = -root3;
                s[ACCESS(7,1,nip,3)] = -root3;
                s[ACCESS(7,2,nip,3)] = -root3;
                for ( i = 0; i < nip; i++)
                    wt[i] = 1.0;
                break;
            case 14:
                b = 0.795822426;
                c = 0.758786911;
                for  (i = 0; i < nip; i++) {
                    if (7 > i)
                        wt[i]=0.886426593;
                    else
                        wt[i] = 0.335180055;
                }

                s[ACCESS(0,0,nip,3)] = -b;
                s[ACCESS(1,0,nip,3)] = b;
                s[ACCESS(2,1,nip,3)] = -b;
                s[ACCESS(3,1,nip,3)] = b;
                s[ACCESS(4,2,nip,3)] = -b;
                s[ACCESS(5,2,nip,3)] = b;

                for  (i = 6; i < nip; i++) {
                    s[ACCESS(i,0,nip,3)] = c;
                    s[ACCESS(i,1,nip,3)] = c;
                    s[ACCESS(i,2,nip,3)] = c;
                }
                s[ACCESS(6,0,nip,3)] = -c;
                s[ACCESS(6,1,nip,3)] = -c;
                s[ACCESS(6,2,nip,3)] = -c;
                s[ACCESS(7,1,nip,3)] = -c;
                s[ACCESS(7,2,nip,3)] = -c;
                s[ACCESS(8,0,nip,3)] = -c;
                s[ACCESS(8,2,nip,3)] = -c;
                s[ACCESS(9,2,nip,3)] = -c;
                s[ACCESS(10,0,nip,3)] = -c;
                s[ACCESS(10,1,nip,3)] = -c;
                s[ACCESS(11,1,nip,3)] = -c;
                s[ACCESS(12,0,nip,3)] = -c;
                break;
            case 15:
                b = 1.0;
                c = 0.674199862;
                wt[0] = 1.564444444;
                for  (i = 1; i < nip; i++) {
                    if (8 > i)
                        wt[i]=0.355555556;
                    else
                        wt[i] = 0.537777778;
                }
                s[ACCESS(1,0,nip,3)] = -b;
                s[ACCESS(2,0,nip,3)] = b;
                s[ACCESS(3,1,nip,3)] = -b;
                s[ACCESS(4,1,nip,3)] = b;
                s[ACCESS(5,2,nip,3)] = -b;
                s[ACCESS(6,2,nip,3)] = b;
                for  (i = 7; i < nip; i++) {
                    s[ACCESS(i,0,nip,3)] = c;
                    s[ACCESS(i,1,nip,3)] = c;
                    s[ACCESS(i,2,nip,3)] = c;
                }
                s[ACCESS(7,0,nip,3)] = -c;
                s[ACCESS(7,1,nip,3)] = -c;
                s[ACCESS(7,2,nip,3)] = -c;
                s[ACCESS(8,1,nip,3)] = -c;
                s[ACCESS(8,2,nip,3)] = -c;
                s[ACCESS(9,0,nip,3)] = -c;
                s[ACCESS(9,2,nip,3)] = -c;
                s[ACCESS(10,2,nip,3)] = -c;
                s[ACCESS(11,0,nip,3)] = -c;
                s[ACCESS(11,1,nip,3)] = -c;
                s[ACCESS(12,1,nip,3)] = -c;
                s[ACCESS(13,0,nip,3)] = -c;
                break;
            case 27:
                for (int j= 0; j <3; j++) {
                    for (i =0; i<9; i++) {
                        wt[9*j +i]=w[j]*v[i];
                    }
                }
                for  (i = 0; i < 25; i=i+3)
                    s[ACCESS(i,0,nip,3)] = -r15;
                for  (i = 1; i < 26; i=i+3)
                    s[ACCESS(i,0,nip,3)] = 0.0;
                for  (i = 2; i < 27; i=i+3)
                        s[ACCESS(i,0,nip,3)] = r15;

                for (int j=0; j < 27;j=j+9) {
                    for  (i = j; i < j+3; i++)
                        s[ACCESS(i,2,nip,3)] = r15;
                    for  (i = j+3; i < j+6; i++)
                        s[ACCESS(i,2,nip,3)] = 0;
                    for  (i = j+6;i < j+9; i++)
                        s[ACCESS(i,2,nip,3)] = -r15;
                }

                for  (i = 0; i < 9; i++)
                    s[ACCESS(i,1,nip,3)] = -r15;
                for  (i = 9; i < 18; i++)
                    s[ACCESS(i,1,nip,3)] = 0;
                for  (i = 18; i < 27; i++)
                    s[ACCESS(i,1,nip,3)] = r15;
                break;
            default:
                std::cout << " !!!!!! WARNING: Wrong number of integrating points for a hexahedron (subroutine sample) !!!!! \n";
        }

    }
    return;
}

void inline shape_der(double * der, double *points, int ipoint, int nod, int ndim, int nip){
    double xi, eta, zeta, etam,xim,zetam,xip,etap,zetap;
    if ((nod != 8)||(ndim != 3)) {
        std::cout << " !!!!!! n != 3 nod != 8 NOT IMPLEMENTED IN FUNCTION shape_der IN __FILE__ LINE __LINE__ !!!!! \n";
        return;
    }
    xi = points[ACCESS(ipoint,0,nip,ndim)];
    eta = points[ACCESS(ipoint,1,nip,ndim)];
    zeta = points[ACCESS(ipoint,2,nip,ndim)];
    etam = 1.0-eta;
    xim = 1.0-xi;
    zetam = 1.0-zeta;
    etap = eta+1.0;
    xip = xi+1.0;
    zetap = zeta+1.0;

    der[ACCESS(0,0,ndim,nod)] = -0.125 *etam *zetam;
    der[ACCESS(0,1,ndim,nod)] = -0.125 *etam *zetap;
    der[ACCESS(0,2,ndim,nod)] = 0.125 *etam *zetap;
    der[ACCESS(0,3,ndim,nod)] = 0.125 *etam *zetam;
    der[ACCESS(0,4,ndim,nod)] = -0.125 *etap *zetam;
    der[ACCESS(0,5,ndim,nod)] = -0.125 *etap *zetap;
    der[ACCESS(0,6,ndim,nod)] = 0.125 *etap *zetap;
    der[ACCESS(0,7,ndim,nod)] = 0.125 *etap *zetam;

    der[ACCESS(1,0,ndim,nod)] = -0.125 *xim *zetam;
    der[ACCESS(1,1,ndim,nod)] = -0.125 *xim *zetap;
    der[ACCESS(1,2,ndim,nod)] = -0.125 *xip *zetap;
    der[ACCESS(1,3,ndim,nod)] = -0.125 *xip *zetam;
    der[ACCESS(1,4,ndim,nod)] = 0.125 *xim *zetam;
    der[ACCESS(1,5,ndim,nod)] = 0.125 *xim *zetap;
    der[ACCESS(1,6,ndim,nod)] = 0.125 *xip *zetap;
    der[ACCESS(1,7,ndim,nod)] = 0.125 *xip *zetam;

    der[ACCESS(2,0,ndim,nod)] = -0.125 *xim *etam;
    der[ACCESS(2,1,ndim,nod)] = 0.125 *xim *etam;
    der[ACCESS(2,2,ndim,nod)] = 0.125 *xip *etam;
    der[ACCESS(2,3,ndim,nod)] = -0.125 *xip *etam;
    der[ACCESS(2,4,ndim,nod)] = -0.125 *xim *etap;
    der[ACCESS(2,5,ndim,nod)] = 0.125 *xim *etap;
    der[ACCESS(2,6,ndim,nod)] = 0.125 *xip *etap;
    der[ACCESS(2,7,ndim,nod)] = -0.125 *xip *etap;

    return;
}

double inline determinant(double* a,int n) {
    if ( 1 == n)
        return a[0];
    else if ( 2 == n )
        return (a[ACCESS(0,0,2,2)]*a[ACCESS(1,1,2,2)] -a[ACCESS(1,0,2,2)]*a[ACCESS(0,1,2,2)]);
    else if ( 3 == n) {//von Saurrus
        return (a[ACCESS(0,0,3,3)]*a[ACCESS(1,1,3,3)]*a[ACCESS(2,2,3,3)]
                +a[ACCESS(0,1,3,3)]*a[ACCESS(1,2,3,3)]*a[ACCESS(2,0,3,3)]
                +a[ACCESS(0,2,3,3)]*a[ACCESS(1,0,3,3)]*a[ACCESS(2,1,3,3)]
                -a[ACCESS(0,2,3,3)]*a[ACCESS(1,1,3,3)]*a[ACCESS(2,0,3,3)]
                -a[ACCESS(0,0,3,3)]*a[ACCESS(1,2,3,3)]*a[ACCESS(2,1,3,3)]
                -a[ACCESS(0,1,3,3)]*a[ACCESS(1,0,3,3)]*a[ACCESS(2,2,3,3)]);
    }
    else {
        std::cout << " !!!!!! n > 3 NOT IMPLEMENTED IN FUNCTION determinant IN __FILE__ LINE __LINE__ !!!!! \n";
        return -666.666;
    }
}

void inline invert(double *a, double *b, int n){
    double idet = 1/determinant(a,n);
    if ( 1 == n) {
        b[0]=1/a[0];
        return;
    }
    else if ( 2 == n)
    {
        return;
    }
    else if ( 3 == n) {
        b[ACCESS(0,0,3,3)] = idet * (a[ACCESS(1,1,3,3)]*a[ACCESS(2,2,3,3)] - a[ACCESS(2,1,3,3)]*a[ACCESS(1,2,3,3)]);
        b[ACCESS(1,0,3,3)] = idet * (a[ACCESS(2,0,3,3)]*a[ACCESS(1,2,3,3)] - a[ACCESS(1,0,3,3)]*a[ACCESS(2,2,3,3)]);
        b[ACCESS(2,0,3,3)] = idet * (a[ACCESS(1,0,3,3)]*a[ACCESS(2,1,3,3)] - a[ACCESS(2,0,3,3)]*a[ACCESS(1,1,3,3)]);
        b[ACCESS(0,1,3,3)] = idet * (a[ACCESS(2,1,3,3)]*a[ACCESS(0,2,3,3)] - a[ACCESS(0,1,3,3)]*a[ACCESS(2,2,3,3)]);
        b[ACCESS(1,1,3,3)] = idet * (a[ACCESS(0,0,3,3)]*a[ACCESS(2,2,3,3)] - a[ACCESS(2,0,3,3)]*a[ACCESS(0,2,3,3)]);
        b[ACCESS(2,1,3,3)] = idet * (a[ACCESS(2,0,3,3)]*a[ACCESS(0,1,3,3)] - a[ACCESS(0,0,3,3)]*a[ACCESS(2,1,3,3)]);
        b[ACCESS(0,2,3,3)] = idet * (a[ACCESS(0,1,3,3)]*a[ACCESS(1,2,3,3)] - a[ACCESS(1,1,3,3)]*a[ACCESS(0,2,3,3)]);
        b[ACCESS(1,2,3,3)] = idet * (a[ACCESS(1,0,3,3)]*a[ACCESS(0,2,3,3)] - a[ACCESS(0,0,3,3)]*a[ACCESS(1,2,3,3)]);
        b[ACCESS(2,2,3,3)] = idet * (a[ACCESS(0,0,3,3)]*a[ACCESS(1,1,3,3)] - a[ACCESS(1,0,3,3)]*a[ACCESS(0,1,3,3)]);
        return;
    }
    else {
        std::cout << " !!!!!! n > 3 NOT IMPLEMENTED IN FUNCTION inverse IN __FILE__ LINE __LINE__ !!!!! \n";
        return;
    }
}

void inline b_mat(double *B, double *deriv, int nod,int ndof,int nst) {
    int l,k,n;
    double x,y,z;
    for (int i = 0;i < nst * ndof; i++)
        B[i]=0;

    for (int i = 0; i < nod ; i++) {
        l = 3*i;
        k = l+1;
        n = l+2;
        x = deriv[ACCESS(0,i,3,nod)];
        y = deriv[ACCESS(1,i,3,nod)];
        z = deriv[ACCESS(2,i,3,nod)];//TODO: to check the row
        B[ACCESS(0,l,6,ndof)] = x;
        B[ACCESS(3,k,6,ndof)] = x;
        B[ACCESS(5,n,6,ndof)] = x;
        B[ACCESS(1,k,6,ndof)] = y;
        B[ACCESS(3,l,6,ndof)] = y;
        B[ACCESS(4,n,6,ndof)] = y;
        B[ACCESS(2,n,6,ndof)] = z;
        B[ACCESS(4,k,6,ndof)] = z;
        B[ACCESS(5,l,6,ndof)] = z;
    }
}

void inline aXpY(double alpha, double *x, double *y,int n, int dim){
    int i;
    if (1 == n) //sum up
        for (i = 0; i < dim; i++)
            y[i] = y[i] + x[i]*alpha;
    else //dont sum up
        for (i = 0; i < dim; i++)
            y[i] = x[i]*alpha;
}

void inline invar(double *stress, double &sigma, double &dsbar, double &theta, int nst) {
    double sx,sy,sz,txy,dx,dy,dz,xj3,sine,s_11,s_22,s_33,s_12,s_13,s_23,ds1,ds2,ds3,d2,d3,sq3;
    switch (nst) {
        case 4:
            sx = stress[0];
            sy = stress[1];
            txy = stress[2];
            sz = stress[3];
            sigma = (sx+sy+sz)/3.0;
            dsbar = sqrt((sx-sy)*(sx-sy)+(sy-sz)*(sy-sz)+(sz-sx)*(sz-sx)+6.0*txy*txy)/sqrt(2.0);
            if (dsbar<1.e-10)
                theta = .0;
            else
            {
                dx = (2.0*sx-sy-sz)/3.0;
                dy = (2.0*sy-sz-sx)/3.0;
                dz = (2.0*sz-sx-sy)/3.0;
                xj3 = dx*dy*dz-dz*txy*txy;
                sine = -13.5*xj3/(dsbar*dsbar*dsbar);
                if (sine>1.)
                    sine = 1.0;
                if (sine<-1.)
                    sine = -1.0;
                theta = asin(sine)/3.0;
            }
            break;
        case 6:
            sq3 = sqrt(3.0);
            s_11 = stress[0];
            s_22 = stress[1];
            s_33 = stress[2];
            s_12 = stress[3];
            s_13 = stress[4];
            s_23 = stress[5];
            sigma = (s_11+s_22+s_33)/3.0;
            d2 = ((s_11-s_22)*(s_11-s_22)+(s_22-s_33)*(s_22-s_33)+(s_33-s_11)*(s_33-s_11))/6.0+s_12*s_12+s_13*s_13+s_23*s_23;
            ds1 = s_11-sigma;
            ds2 = s_22-sigma;
            ds3 = s_33-sigma;
            d3 = ds1*ds2*ds3-ds1*s_13*s_13-ds2*s_23*s_23-ds3*s_12*s_12+2.0*s_12*s_13*s_23;
            dsbar = sq3*sqrt(d2);
            if (dsbar == 0.)
                theta = 0.0;
            else {
                sine = -3.0*sq3*d3/(2.0*d2*sqrt(d2));
                if (sine>1.)
                    sine = 1.0;
                if (sine<-1.)
                    sine = -1.0;
                theta = asin(sine)/3.0;
            }
            break;
        default:
            std::cout << "WARNING: Wrong size of stiffnessmatrix (subroutine invar) \n";
    }
    return;
}

#ifdef SELFTEST
int main() {
    //double a[9] = {1.0,2.2,2.0,2.3,23.0,32.3,3.0,3.4,4.0};
    double a[9] ={1.0,2.0,2.0,1.0,2.0,2.01,1.0,2.5,2.01};
    //double a[9]={1,2,2,2,2,3,3,5,4};
    double b[9],c[9];
    double det;
    cout << "a: ";
    for (int i = 0;i < 9; i++) {
        cout << a[i] << " ";
        if ((i % 3) == 2)
            cout << endl << "   ";
    }
    cout << endl;
    for (int i = 0;i < 3; i++) {
        for (int j = 0;j < 3; j++)
            cout << a[ACCESS(i,j,3,3)] << " ";
        cout << endl;
    }

    det = determinant(a, 3);
    cout << "det(a): " << det << endl;
    invert(a,b,3);
    matmulAB(a,3,3,b,3,3,c,3,3);
    cout << "c: ";
    for (int i = 0;i < 9; i++) {
        cout << setiosflags(ios::scientific) << c[i] << " ";
        if ((i % 3) == 2)
            cout << endl << "   ";
    }
    cout << endl;
    return 0;
}
#endif
