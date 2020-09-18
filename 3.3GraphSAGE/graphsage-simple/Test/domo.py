class Solution:
    def permuteUnique(self, nums):
        res = []
        self.helper(nums,[],res)
        return res
    def helper(self,left,path,res):
        if len(left)==0:
            res.append(path)
        for i in set(left):
            left.remove(i)
            path+=[i]
            return self.helper(left,path,res)
s = Solution()
nums = [1,1,2]
res = s.permuteUnique(nums)
print(res)