
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "../block2_big_site.hpp"

template struct block2::SZMatrix<double>;
template struct block2::SU2Matrix<double>;

template struct block2::DRT<block2::SZ, block2::ElemOpTypes::SZ>;
template struct block2::HDRT<block2::SZ, block2::ElemOpTypes::SZ>;

template struct block2::HDRTScheme<block2::SZ, double,
                                   block2::ElemOpTypes::SZ>;
template struct block2::DRTBigSiteBase<block2::SZ, double>;
template struct block2::DRTBigSite<block2::SZ, double>;

template struct block2::DRT<block2::SU2, block2::ElemOpTypes::SU2>;
template struct block2::HDRT<block2::SU2, block2::ElemOpTypes::SU2>;

template struct block2::HDRTScheme<block2::SU2, double,
                                   block2::ElemOpTypes::SU2>;
template struct block2::DRTBigSiteBase<block2::SU2, double>;
template struct block2::DRTBigSite<block2::SU2, double>;
