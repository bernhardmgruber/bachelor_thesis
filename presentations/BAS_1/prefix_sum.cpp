int main()
{
	int arr[13] = {5, 1, 2, 4, 8, 2, 8, 1, 4, 8, 5, 2, 5};
	int sum[13];
	
sum[0] = arr[0];
for(int i = 1; i < length(arr); i++)
	sum[i] = sum[i - 1] + arr[i];
	
	return 0;
}