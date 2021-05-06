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

open REF, "../../Tapping_activity_training.tsv" or die;
while ($line=<REF>){
	$line=~s/"//g;
	chomp $line;
	@table=split "\t", $line;
	$map{$table[2]}=$table[3];
}
close REF;

open FILE, "test_gs.dat" or die;
$i=0;
while ($line=<FILE>){
	chomp $line;
	@table=split "\t", $line;
#	print FINAL "$table[1]\t";
	$map_gs{$map{$table[0]}}=$table[1];

	if ($ref[$i] eq ""){
#		print FINAL "$avg";
		$map_value{$map{$table[0]}}+=$avg;
		$map_count{$map{$table[0]}}++;
	}else{
		$val=$ref[$i]/$ref_count[$i];
		$map_value{$map{$table[0]}}+=$val;
		$map_count{$map{$table[0]}}++;
		#	print FINAL "$val";
	}

	print FINAL "\n";
	$i++;
}
close FINAL;
close FILE;

open FINAL, ">feature_test.txt" or die;
@all=keys %map_count;
foreach $pid (@all){
	print FINAL "$map_gs{$pid}";
	$val=$map_value{$pid}/$map_count{$pid};
	print FINAL "\t$val\n";
}
close FINAL;
