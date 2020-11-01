class Solution:
    def islandPerimeter(self, grid) -> int:
        c = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            grid[i].insert(n, 0)
            grid[i].insert(0, 0)
        grid.insert(m, [0] * (n + 2))
        grid.insert(0, [0] * (n + 2))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if grid[i][j] == 1:
                    c += [grid[i - 1][j], grid[i + 1][j], grid[i][j - 1], grid[i][j + 1]].count(0)
        return c



if __name__ == "__main__":
    grid = [[0, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 1, 0, 0],
     [1, 1, 0, 0]]
    s = Solution()
    res = s.islandPerimeter(grid)
    print(res)