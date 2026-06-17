
/** @file complex_d.h
 *
 * Interface to complex numbers 
 *
 */

/*
 *  Copyright (C) 2002 Stefan Weinzierl
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef __PDF_COMPLEX_D_H__
#define __PDF_COMPLEX_D_H__

#include <complex>

namespace pdf {

  /// complex numbers with double precision
  typedef std::complex<double> complex_d;
  typedef std::complex<long double> complex_ld;
  typedef long double ldouble;


} // namespace pdf

#endif // ndef __PDF_COMPLEX_D_H__

