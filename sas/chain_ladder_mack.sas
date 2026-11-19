/* Chain ladder and Mack (1993) standard error on the Taylor and Ashe (1983) triangle.
   Base SAS only (DATA step arrays, PROC PRINT). Same formulas as src/reserving/methods.py.
   Expected: chain ladder reserve 18,680,856; Mack standard error of the total 2,447,095. */

data work.ta_incr;                       /* incremental paid claims, accident years 1 to 10 */
  input ay d1-d10;
  datalines;
1 357848 766940 610542 482940 527326 574398 146342 139950 227229 67948
2 352118 884021 933894 1183289 445745 320996 527804 266172 425046 .
3 290507 1001799 926219 1016654 750816 146923 495992 280405 . .
4 310608 1108250 776189 1562400 272482 352053 206286 . . .
5 443160 693190 991983 769488 504851 470639 . . . .
6 396132 937085 847498 805037 705960 . . . . .
7 440832 847631 1131398 1063269 . . . . . .
8 359480 1061648 1443370 . . . . . . .
9 376686 986608 . . . . . . . .
10 344014 . . . . . . . . .
;
run;

data work.ta_cum;                        /* cumulative triangle */
  set work.ta_incr;
  array d{10} d1-d10;
  array c{10} c1-c10;
  retain s 0;
  s = 0;
  do j = 1 to 10;
    if d{j} ne . then do; s + d{j}; c{j} = s; end;
    else c{j} = .;
  end;
  keep ay c1-c10;
run;

data work.cl_mack;
  set work.ta_cum end=eof;
  array c{10, 10} _temporary_;
  array f{9}      _temporary_;
  array s2{9}     _temporary_;
  array cs{9}     _temporary_;           /* column sums over rows that have the next column */
  array ult{10}   _temporary_;
  array mse{10}   _temporary_;
  array cc{10} c1-c10;
  n = 10;
  do j = 1 to n; c{ay, j} = cc{j}; end;

  if eof then do;
    /* volume-weighted age-to-age factors and column sums */
    do k = 1 to n - 1;
      num = 0; den = 0;
      do i = 1 to n - k;
        num + c{i, k + 1}; den + c{i, k};
      end;
      f{k} = num / den; cs{k} = den;
    end;
    /* Mack variance parameters sigma^2 */
    do k = 1 to n - 2;
      s = 0;
      do i = 1 to n - k;
        s + c{i, k} * (c{i, k + 1} / c{i, k} - f{k}) ** 2;
      end;
      s2{k} = s / (n - k - 1);
    end;
    s2{n - 1} = min(s2{n - 2} ** 2 / s2{n - 3}, min(s2{n - 3}, s2{n - 2}));

    /* ultimates and reserves */
    total_res = 0; total_var = 0;
    do i = 1 to n;
      a = n - i + 1;                       /* column of the latest diagonal */
      u = c{i, a};
      do k = a to n - 1; u = u * f{k}; end;
      ult{i} = u;
      mse{i} = 0;
      if i >= 2 then do;
        sm = 0;
        do k = a to n - 1;
          ck = c{i, a};
          do q = a to k - 1; ck = ck * f{q}; end;
          sm + s2{k} / f{k} ** 2 * (1 / ck + 1 / cs{k});
        end;
        mse{i} = u ** 2 * sm;
      end;
      total_res + (u - c{i, a});
      total_var + mse{i};
    end;
    /* covariance between accident years */
    do i = 2 to n - 1;
      a = n - i + 1;
      sm = 0;
      do k = a to n - 1; sm + s2{k} / f{k} ** 2 / cs{k}; end;
      do j = i + 1 to n;
        total_var + 2 * ult{i} * ult{j} * sm;
      end;
    end;

    /* output by accident year and total */
    do i = 1 to n;
      a = n - i + 1;
      ay = i; latest = c{i, a}; ultimate = ult{i}; reserve = ult{i} - c{i, a};
      se = sqrt(mse{i});
      output;
    end;
    ay = .; latest = .; ultimate = .; reserve = total_res; se = sqrt(total_var);
    output;
  end;
  keep ay latest ultimate reserve se;
run;

title "Taylor and Ashe (1983): chain ladder reserve and Mack standard error";
proc print data=work.cl_mack noobs; format latest ultimate reserve se comma16.; run;
