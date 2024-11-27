import { Model } from './Model';

export class NewModel extends Model {
    // 定义模型的属性
    private name: string;
    private description: string;

    constructor(name: string, description: string) {
        super();
        this.name = name;
        this.description = description;
    }

    // 获取名称
    getName(): string {
        return this.name;
    }

    // 设置名称
    setName(name: string): void {
        this.name = name;
    }

    // 获取描述
    getDescription(): string {
        return this.description;
    }

    // 设置描述
    setDescription(description: string): void {
        this.description = description;
    }

    // 实现必要的模型方法
    toJSON(): object {
        return {
            name: this.name,
            description: this.description
        };
    }

    // 从JSON创建模型实例
    static fromJSON(json: any): NewModel {
        return new NewModel(json.name, json.description);
    }
} 