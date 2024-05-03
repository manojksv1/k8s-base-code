resource "aws_route_table" "db-route-table" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name  = "DB-route"
    from  = "trrafrom"
    owner = "manoj"
  }
}

resource "aws_route_table_association" "association_db" {
  route_table_id = aws_route_table.db-route-table.id
  count          = length(var.db-cidr)
  subnet_id      = aws_subnet.DB-subnet[count.index].id
}