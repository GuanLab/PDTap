#!/usr/bin/perl
#
$fold=$ARGV[0];
@mat=glob("fold_${fold}/eva.txt*test");
$total_file=0;
$total=0;
$total_c=0;
foreach $eva (@mat){
	open PRED, $eva or die;
	$i=0;
	while ($line=<PRED>){
		chomp $line;
		@table=split "\t", $line;
		if ($table[1] eq ""){}else{
			$ref[$i]+=$table[1];
			$ref_count[$i]++;
			$total+=$table[1];
			$total_c++;
		}
		$val[$i]=$table[0];
		$i++;
	}
	close PRED;
	$total_file++;
}
$avg=$total/$total_c;


open FINAL, ">feature_test.txt" or die;
open FILE, "test_gs.dat" or die;
$i=0;
while ($line=<FILE>){
	chomp $line;
	@table=split "\t", $line;
	print FINAL "$table[1]\t";

	if ($ref[$i] eq ""){
		print FINAL "$avg";
	}else{
		$val=$ref[$i]/$ref_count[$i];
		print FINAL "$val";
	}

	print FINAL "\n";
	$i++;
}
close FINAL;
close FILE;

